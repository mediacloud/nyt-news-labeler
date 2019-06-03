# coding=utf-8

import unittest

import models


class ModelsTestCase(unittest.TestCase):
    # https://globalvoices.org/specialcoverage/a-duty-to-remember-30-years-after-tiananmen/
    SAMPLE_TEXT = u"""
        A duty to remember: 30 years after Tiananmen
        ---

        It has been 30 years since the rise and fall of the 89 Democracy Movement (八九民运) in China that culminated in
        the infamous Tiananmen Square Massacre on June 4, 1989.

        On that day, the Chinese military carried out a brutal crackdown on student-led demonstrations calling for
        democratic reforms. The Chinese Red Cross estimated that 2,700 civilians were killed, but other sources point to
        a much higher toll. A confidential US government document unveiled in 2014 reported that a Chinese internal
        assessment estimated that at least 10,454 civilians were killed.

        The Communist Party of China has never publicly acknowledged these events or accounted for its actions with an
        independent investigation. There are no references to the 89 Democracy Movement in any history textbooks and
        most university students in China have never heard about the massacre.

        Global Voices has been covering the issue for over a decade. This year we commemorate the 30th anniversary of
        what led to the June 4 massacre to fulfill our duty to keep the memory of those events alive, despite continuous
        efforts by Beijing to deny basic historical truth. 
    """

    @classmethod
    def setUpClass(cls):
        models.initialize()

    @staticmethod
    def _labels_set(labels_with_confidence):
        """Return a set of all labels.

        Models return a rather unpredictable list of labels with confidences, so we test whether a list of top labels
        are present somewhere in the list.
        """
        labels = []
        for label in labels_with_confidence:
            label_name, label_confidence = label
            labels.append(label_name)
        return set(labels)

    def test_model600(self):
        res600 = models.model600.predict(self.SAMPLE_TEXT)
        assert {
            u'war and revolution',
            u'demonstrations and riots',
            u'politics and government',
        }.issubset(self._labels_set(res600)), "model600 top labels are present"

    def test_model3000(self):
        res3000 = models.model3000.predict(self.SAMPLE_TEXT)
        assert {
            u'politics and government',
            u'freedom and human rights',
            u'demonstrations and riots',
        }.issubset(self._labels_set(res3000)), "model3000 top labels are present"

    def test_model_all(self):
        res_all = models.model_all.predict(self.SAMPLE_TEXT)
        assert {
            u'politics and government',
            u'freedom and human rights',
            u'demonstrations and riots',
        }.issubset(self._labels_set(res_all)), "model_all top labels are present"

    def test_model_with_tax(self):
        res_with_tax = models.model_with_tax.predict(self.SAMPLE_TEXT)
        assert {
            u'Top/Features/Travel/Guides/Destinations/Asia/China',
            u'Top/Features/Travel/Guides/Destinations/Asia',
            u'Top/Opinion/Opinion',
        }.issubset(self._labels_set(res_with_tax)), "model_with_tax top labels are present"

    def test_model_just_tax(self):
        res_just_tax = models.model_just_tax.predict(self.SAMPLE_TEXT)
        assert {
            u'top/features/travel/guides/destinations/asia/china',
            u'top/news/world/countries and territories/china',
            u'top/features/travel/guides/destinations/asia',
        }.issubset(self._labels_set(res_just_tax)), "model_just_tax top labels are present"
