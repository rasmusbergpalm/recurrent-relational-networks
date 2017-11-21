from unittest import TestCase

from tasks.babi.data import bAbI


class TestBAbI(TestCase):

    def test_babi(self):
        data = bAbI('en-valid-10k')
        self.assertEquals(20, len(data.train))
        self.assertEquals(20, len(data.valid))
        self.assertEquals(20, len(data.test))
