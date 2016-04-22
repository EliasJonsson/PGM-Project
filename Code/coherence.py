# -*- coding: utf-8 -*-
"""Module to handle calculating coherence scores."""
from __future__ import absolute_import, unicode_literals, print_function
# Third party modules
import requests

score_types = ['ca', 'cp', 'cv', 'npmi', 'uci', 'umass']
_palmetto_url = "http://palmetto.aksw.org/palmetto-webapp/service/{score}"

def coherence_scores(topic_words):
	"""Function to calculate coherence scores"""
	scores = {}
	for score_type in score_types:
		url = _palmetto_url.format(score=score_type)
		r = requests.get(url, {'words': ' '.join(topic_words)})
		scores[score_type] = float(r.text)

	return scores
