"""
    Copyright 2018 Google LLC
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        https://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

"""Parses boolean expressions (as strings) to produce "calculation graph".

## Propositional logic

Given an expression like "a|((~a)&(b|c))", the function `parse` will output a
list of "op types" (which is variable names, or logical operands), and the list
of indices (relative to current op) that feed in to each operand. This is done
in an order that allows computation of the boolean expression. For example, the
above expression becomes:

(position):   OPS:   INPUTS:
0             a      []
1             a      []
2             ~      [-1]
3             b      []
4             c      []
5             |      [-2, -1]
6             &      [-4, -1]
7             |      [-7, -1]

## First-order logic

The above is also extended to first-order logic, with relations, "for all", and
"exists". For example, 'f(x, y)' is a relation, and 'all x . (f(x) -> g(x))' is
a "for all" formula.

Unary and binary relations are currently supported in this model. A binary
relation f(x, y) is parsed as a ternary op, with op-type "R2", and arguments
[f, x, y].

For all "all x . k" is parsed as a binary op, with op-type "A" and arguments
[x, k]. Similarly for "exists x . k".
"""

import collections
import functools

import pyparsing


class Language(object):
    """Contains information on a language (e.g., propn. or first-order logic).

    This is used by `TreeNet` to learn how to interpret the symbols, and how many
    "variable" embeddings to create
    """

    def __init__(self, arities, predicates, constants, variables):
        """Initializes a `Language`.

        Args:
          arities: `OrderedDict` containing a mapping from op symbols to arities.
          predicates: List of strings, containing valid predicate (or relation)
              symbols.
          constants: List of strings, containing valid FOL constant symbols.
          variables: List of strings, containing valid FOL variable symbols. Note
              that "propositional variables" should be specified as (zero arity)
              predicates.
        """
        assert isinstance(arities, collections.OrderedDict)
        self._arities = arities
        self._predicates = predicates
        self._constants = constants
        self._variables = variables
        self._max_arity = max(arities.values())
        self._ops = list(arities.keys())
        self._symbols = list(arities.keys()) + predicates + constants + variables

    @property
    def arities(self):
        return self._arities

    @property
    def predicates(self):
        return self._predicates

    @property
    def constants(self):
        return self._constants

    @property
    def variables(self):
        return self._variables

    def arity(self, op):
        if op in self._predicates or op in self._constants or op in self._variables:
            return 0
        else:
            return self._arities[op]

    @property
    def max_arity(self):
        return self._max_arity

    @property
    def ops(self):
        return self._ops

    @property
    def symbols(self):
        """Returns ops and variables."""
        return self._symbols


# These strings are used in internal representations of the ops when parsed, and
# are stored in sstables when parsing the text data, and then cross-referenced
# when understanding the ops in the TreeNet code. (E.g., important to be able to
# distinguish unary and binary ops.) I.e., if these get changed, then data
# generation likely has to be done again.
#
# This isn't the same as the input operations allowed - there can be a many-to-
# one mapping in this case. E.g., both /\ and & are recognised for AND.

IDENTITY_SYMBOL = ''
NEGATION_SYMBOL = '~'
AND_SYMBOL = '&'
OR_SYMBOL = '|'
XOR_SYMBOL = '^'
IMPLIES_SYMBOL = '>'
FOR_ALL_SYMBOL = 'A'
EXISTS_SYMBOL = 'E'
RELATION_SYMBOL = 'R{}'  # formatted for arity of relation.
FALSE_SYMBOL = 'F'
TRUE_SYMBOL = 'T'


def propositional_language(num_variables=26):
    """Makes a propositional logic language."""
    predicates = [chr(ord('a') + i) for i in range(num_variables)]

    return Language(
        collections.OrderedDict([
            (IDENTITY_SYMBOL, 0),
            (NEGATION_SYMBOL, 1),
            (AND_SYMBOL, 2),
            (OR_SYMBOL, 2),
            (XOR_SYMBOL, 2),
            (IMPLIES_SYMBOL, 2),
        ]),
        predicates=predicates,
        constants=[],
        variables=[],
    )


FOL_MAX_RELATION_ARITY = 2


def fol_language():
    """Makes a first-order logic language.

    This has:

    *   Predicate symbols p1, ..., p9, q1, ..., r9.
    *   Constant symbols a1, ..., a9, b1, ..., c9.
    *   Variable symbols x1, ..., x9, y1, ..., z9.

    Returns:
      Instance of `Language`.
    """

    def make_symbols(start):
        """E.g., if start='a', then returns ['a1', ..., 'a9', 'b1', ..., 'c9']."""
        return [chr(ord(start) + i) + str(n)
                for i in range(0, 3)
                for n in range(1, 10)]

    return Language(
        collections.OrderedDict([
            (IDENTITY_SYMBOL, 0),
            (NEGATION_SYMBOL, 1),
            (AND_SYMBOL, 2),
            (OR_SYMBOL, 2),
            (XOR_SYMBOL, 2),
            (IMPLIES_SYMBOL, 2),
            (FOR_ALL_SYMBOL, 2),
            (EXISTS_SYMBOL, 2),
            (RELATION_SYMBOL.format(1), 2),  # unary-relation
            (RELATION_SYMBOL.format(2), 3),  # binary-relation
        ]),
        predicates=make_symbols('p'),
        constants=make_symbols('a'),
        variables=make_symbols('x'),
    )


# Makes parsing a lot faster:
pyparsing.ParserElement.enablePackrat()


class _SubExpression(
    collections.namedtuple('_SubExpression', ('ops', 'inputs'))):
    """Contains a parsed boolean expression.

    Attributes:
      ops: List of types, which is variable names or operators. For example,
          ['a', 'b', '~', '&'].
      inputs: List of list of input indices relative to the current index (i.e.,
          they are negative numbers).
    """


class ParseResult(
    collections.namedtuple('ParseResult', ('expression', 'ops', 'inputs'))):
    """Final parse output.

    This is like `SubExpression`, but with a couple of extra useful fields. It is
    used when generating datasets, as the fields it contains are in a suitable
    format for writing to SSTables.

    Attributes:
      expression: List of ops (including variable letters and brackets) in the
          original expression order.
      ops: List of ops (including variable letters) for calculating the boolean
          expression.
      inputs: List of list of input indices relative to the current index (i.e.,
          they are negative numbers).
    """


class ExpressionData(
    collections.namedtuple('ExpressionData', ('expression', 'ops', 'inputs'))):
    """Similar to `ParseResult`, but for batches of TF tensors from datasets.

    Attributes:
      expression: String tensor with shape `[batch_size, max_expression_length]`.
      ops: String tensor with shape `[batch_size, max_ops_length]`.
      inputs: Tensor with shape `[batch_size, max_ops_length, max_arity]`.
    """


def _concat_subexpressions(*expressions):
    """Concatenates the types and input indices of the expressions."""
    ops = []
    inputs = []

    for expression in expressions:
        ops += expression.ops
        inputs += expression.inputs

    return _SubExpression(ops, inputs)


def _ensure_subexpression(expression_or_variable):
    if isinstance(expression_or_variable, _SubExpression):
        return expression_or_variable
    return _SubExpression([expression_or_variable], [[]])


class Parser(object):
    """Parser for tree-like expressions."""

    def __init__(self, language):
        """Initializes a `Parser` instance.

        Args:
          language: Instance of `Language`. Used to determine the different
              predicate / constant / variable symbols appearing.
        """
        self._language = language
        predicate_symbol = pyparsing.oneOf(
            language.predicates + [FALSE_SYMBOL, TRUE_SYMBOL])
        constant_symbol = pyparsing.oneOf(language.constants)
        variable_symbol = pyparsing.oneOf(language.variables)

        left_par = pyparsing.Literal('(').suppress()
        right_par = pyparsing.Literal(')').suppress()

        formula = pyparsing.Forward()

        relation_expressions = self._relation_expressions(
            predicate_symbol, pyparsing.Or([constant_symbol, variable_symbol]))

        formula_without_op = pyparsing.Forward()
        negated_formula_without_op = (
                pyparsing.Literal('~').suppress() + formula_without_op)
        negated_formula_without_op.setParseAction(
            lambda args: self._op(args, NEGATION_SYMBOL))
        formula_without_op <<= pyparsing.MatchFirst(
            [left_par + formula + right_par] + relation_expressions
            + [negated_formula_without_op])

        binary_expressions = self._binary_expressions(
            formula_without_op, formula)

        negation = pyparsing.Literal('~').suppress() + formula
        negation.setParseAction(lambda args: self._op(args, NEGATION_SYMBOL))

        for_all = (pyparsing.Literal('all').suppress() + variable_symbol
                   + pyparsing.Literal('.').suppress() + formula)
        for_all.setParseAction(lambda args: self._op(args, FOR_ALL_SYMBOL))

        exists = (pyparsing.Literal('exists').suppress() + variable_symbol
                  + pyparsing.Literal('.').suppress() + formula)
        exists.setParseAction(lambda args: self._op(args, EXISTS_SYMBOL))

        formula <<= pyparsing.MatchFirst(
            binary_expressions + [negation] + [for_all, exists, formula_without_op])

        self._expression = formula

    def _relation_expressions(self, predicate_symbol,
                              variable_or_constant_symbol):
        """Returns list of `pyparsing.Expression` matching relations."""
        expressions = []
        # Relations of various arities.
        for arity in range(1, FOL_MAX_RELATION_ARITY + 1):
            expression = predicate_symbol + pyparsing.Literal('(').suppress()
            for i in range(arity):
                if i > 0:
                    expression += pyparsing.Literal(',').suppress()
                expression += variable_or_constant_symbol
            expression += pyparsing.Literal(')').suppress()
            relation_symbol = RELATION_SYMBOL.format(arity)
            expression.setParseAction(functools.partial(self._op, op=relation_symbol))
            expressions.append(expression)

        # Also match a nullary relation without arguments
        expressions.append(predicate_symbol)
        return expressions

    def _binary_expressions(self, left_formula, right_formula):
        """Returns list of `pyparsing.Expression` for various binary ops."""
        binary_op_symbols = {
            AND_SYMBOL: '& /\\',
            OR_SYMBOL: '| \\/',
            IMPLIES_SYMBOL: '> ->',
            XOR_SYMBOL: '^',
        }
        expressions = []
        for binary_op, op_symbols in binary_op_symbols.items():
            op = left_formula + pyparsing.oneOf(op_symbols).suppress() + right_formula
            op.setParseAction(functools.partial(self._op, op=binary_op))
            expressions.append(op)
        return expressions

    def _op(self, parse_args, op):
        """Returns a new `SubExpression` from the op and parse args.

        Args:
          parse_args: List of parse args, which should be instances of
              `_SubExpression` or strings representing symbols.
          op: String representing the op, e.g., &, A (for all), etc.

        Returns:
          Instance of `_SubExpression`.
        """
        parse_args = [_ensure_subexpression(arg) for arg in parse_args]
        arity = len(parse_args)
        indices = []
        for i in range(arity):
            if i == 0:
                indices = [-1]
            else:
                indices = [indices[0] - len(parse_args[-i].ops)] + indices
        new = _SubExpression([op], [indices])
        all_expressions = parse_args + [new]
        return _concat_subexpressions(*all_expressions)

    def _clean_expression(self, expression):
        r"""Cleans up the expression string to use canonical ops, no spaces.

        E.g., "all x. (y \\/ z)" will become "Ax.(y|z)" (as a list).

        Args:
          expression: String.

        Returns:
          List of characters containing the ops and variable letters in the order
          they occur in `string`.

        Raises:
          ValueError: If the string contains an unrecognised symbol.
        """
        map_ = collections.OrderedDict([
            ('exists', EXISTS_SYMBOL),
            ('all', FOR_ALL_SYMBOL),
            ('\\/', OR_SYMBOL),
            ('/\\', AND_SYMBOL),
            ('->', IMPLIES_SYMBOL),
            ('>>', IMPLIES_SYMBOL),
            ('~', NEGATION_SYMBOL),
            ('&', AND_SYMBOL),
            ('|', OR_SYMBOL),
            ('^', XOR_SYMBOL),
            ('>', IMPLIES_SYMBOL),
            ('T', TRUE_SYMBOL),
            ('F', FALSE_SYMBOL),
            ('(', '('),
            (')', ')'),
            (',', ','),
            ('.', '.'),
            (' ', None),
        ])
        for c in (self._language.predicates + self._language.constants
                  + self._language.variables):
            map_[c] = c
        keyword_lengths = sorted(set(len(keyword) for keyword in map_.keys()),
                                 reverse=True)

        result = []
        i = 0
        while i < len(expression):
            found = False
            for keyword_length in keyword_lengths:
                if i + keyword_length <= len(expression):
                    extracted = expression[i:i + keyword_length]
                    if extracted in map_:
                        conversion = map_[extracted]
                        if conversion is not None:
                            result.append(conversion)
                        i += keyword_length
                        found = True
                        break
            if not found:
                raise ValueError('Unable to clean {} at position {}'
                                 .format(expression, i + 1))

        return result

    @property
    def language(self):
        """Returns `Language` used by this parser."""
        return self._language

    def parse(self, expression):
        """Parses the expression, extracting ops and indices.

        Args:
          expression: The expression as a string.

        Returns:
          Instance of `ParseResult`.
        """
        try:
            parsed = self._expression.parseString(expression)[0]
        except (pyparsing.ParseException, RuntimeError) as e:
            print(('Unable to parse: {0}'.format(expression)))
            raise e
        parsed = _ensure_subexpression(parsed)

        clean_expression = self._clean_expression(expression)

        return ParseResult(
            expression=[op for op in clean_expression],
            ops=[op for op in parsed.ops],
            inputs=parsed.inputs,
        )
