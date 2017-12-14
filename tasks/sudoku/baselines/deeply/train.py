from tasks.sudoku.baselines.deeply.deepmp import SudokuDeeplyLearnedMessages
import trainer

trainer.train(SudokuDeeplyLearnedMessages(False))
