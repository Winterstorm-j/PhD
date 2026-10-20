"""Name matching utilities for approximate person-name comparison.

This module normalizes names into a common structure containing a last name,
first-name tokens, nickname aliases, and optional suffixes such as Jr. or Sr.
It then compares names using a weighted similarity model designed to be tolerant
of common variations in spelling, initials, abbreviations, and transpositions.

The primary entry point is :class:`NameMatcher`, which exposes methods for
single-name comparison and nearest-neighbour lookup across candidate name lists.
"""

# Name Matching Tools
# Tools for matching person names
# -----------------------------------------------------------------------------
# Class included: NameMatcher
# Public methods: match_names(name1, name2)
#                 parse_name(name)
#                 find_closest_names(target_names, other_names)
# -----------------------------------------------------------------------------
# Author: Natalie Ahn
# Last revised: August 2017

import re
import csv
import numpy as np
from math import log
from statistics import mean
from bisect import bisect_left
from nltk.metrics import edit_distance
from jellyfish import jaro_winkler_similarity

class NameMatcher:
	"""Compare person names using a weighted similarity metric.

	The matcher parses each supplied name into a normalized representation,
	including last name, first-name components, nicknames, and suffixes. It then
	computes a similarity score between 0 and 1 using either a log-transformed
	Levenshtein distance or Jaro-Winkler similarity.
	"""

	suffixes = {'junior', 'jr', 'senior', 'sr', 'ii', 'iii', 'iv'}
	weight_order = ['first_names', 'last_name', 'suffix']

	def __init__(self, distfun='levenshtein'):
		"""Initialize the matcher with a distance function.

		Expected input:
		    distfun: Either ``'levenshtein'`` (default), ``'jaro_winkler'``, or a
		        callable that accepts two strings and returns a normalized distance.

		Expected output:
		    Sets ``self.distfun`` to the selected comparison function and stores the
		    internal tuning parameters used by the matching logic.
		"""
		if distfun == 'levenshtein':
			self.distfun = self._levenshtein_log
		elif distfun == 'jaro_winkler':
			self.distfun = lambda str1,str2: 1 - jaro_winkler_similarity(str1, str2)
		else: self.distfun = distfun
		# The following parameters have been optimized on a limited initial dataset.
		# Users may wish to adjust and experiment with them on their own dataset.
		# Detailed parameter explanations will be provided in the README file.
		self.params = {'weights':[0.25, 0.66, 0.09],
					  'disc_abbrev':0.95, 'disc_abbrev_notstart':0.6,
					  'disc_missing_fname':0.7, 'disc_missing_mname':0.95,
					  'disc_missing_nickname':0.8, 'disc_initial':0.6,
					  'disc_initial_mismatch':0.4, 'disc_acronym':0.5}

	def _levenshtein_pct(self, str1, str2):
		"""Return the normalized Levenshtein distance as a percentage.

		This is a simple edit-distance ratio where 0 means exact agreement and
		values near 1 indicate large differences.
		"""
		return edit_distance(str1, str2) / max(len(str1),len(str2))

	def _levenshtein_log(self, str1, str2):
		"""Return a logarithmic edit-distance score suitable for similarity.

		The score is near 1 for similar strings and decreases as the number of
		edits grows relative to the string length.
		"""
		edits = edit_distance(str1, str2)
		log_edits = 1 - log(max(len(str1),len(str2)) - edits + 1,
							max(len(str1),len(str2)) + 1)
		return log_edits

	def match_names(self, name1, name2, speed='fast', min_last_sim=0.95):
		"""Return a similarity score for two input names.

		The score combines the similarity of the suffix, last name, and first
		name components. When ``speed='fast'`` and the last-name similarity is too
		low, the method short-circuits to avoid performing the more expensive
		first-name comparison.

		Expected input:
		    name1: A name string such as ``'John Smith'`` or ``'Smith, John Jr.'``.
		    name2: A second name string to compare against ``name1``.
		    speed: Either ``'fast'`` for a quick comparison or ``'slow'`` for a
		        more detailed first-name analysis.
		    min_last_sim: Minimum last-name similarity threshold used when
		        ``speed='fast'``.

		Expected output:
		    A float similarity score between approximately 0 and 1. Values closer
		    to 1 indicate stronger agreement between the two names.
		"""
		name_dict1 = self.parse_name(name1.lower())
		name_dict2 = self.parse_name(name2.lower())
		sims = {}
		sims['suffix'] = self._suffix_sim(name_dict1['suffix'], name_dict2['suffix'])
		sims['last_name'] = 1 - self.distfun(name_dict1['last_name'], name_dict2['last_name'])
		if speed == 'fast' and sims['last_name'] < min_last_sim:
			sims['first_names'] = 1 - self.distfun(' '.join(name_dict1['last_name']),
												   ' '.join(name_dict2['last_name']))
			return self._weighted_sum(sims)
		jr1 = self._suffix_acron_letter(name_dict1['suffix'])
		jr2 = self._suffix_acron_letter(name_dict2['suffix'])
		version_sims = []
		for fnames1 in [name_dict1[key] for key in ['first_names', 'nicknames']]:
			for fnames2 in [name_dict2[key] for key in ['first_names', 'nicknames']]:
				version_sims.append(self._max_subseq_sim(fnames1, fnames2, jr1, jr2))
		sims['first_names'] = max(version_sims)
		if (name_dict1['nicknames'] and not name_dict2['nicknames']) \
		or (name_dict2['nicknames'] and not name_dict1['nicknames']):
			sims['first_names'] *= self.params['disc_missing_nickname']
		return self._weighted_sum(sims)

	def parse_name(self, name):
		"""Normalize a single name string into a structured dictionary.

		Expected input:
		    name: A raw name string such as ``'John A. Smith, Jr.'`` or
		        ``'Smith, John "Jack"'``.

		Expected output:
		    A dictionary with keys:
		        - ``first_names``: list of given names or initials
		        - ``last_name``: the surname portion
		        - ``nicknames``: any quoted or bracketed aliases
		        - ``suffix``: a suffix such as ``Jr.``, ``Sr.``, ``II``, etc.
		"""
		name_dict = {'first_names':[], 'last_name':'', 'nicknames':[]}
		last_tok = re.split(r'[,\s]', name)[-1]
		if last_tok.lower().replace('.', '') in self.suffixes:
			name_dict['suffix'] = last_tok
			name = name[:-len(last_tok)].strip()
		else: name_dict['suffix'] = ''
		if re.search(',', name):
			parts = re.split(',', name)
			if len(parts) > 1:
				name_dict['last_name'] = parts[0]
				name = ' '.join(parts[1:]).strip()
		parts = [x for x in re.split('[ .]', name) if x]
		if parts and not name_dict['last_name']:
			last_name_start = len(parts) - 1
			last_name_prefixes = {
				'van', 'von', 'de', 'di', 'da', 'le', 'del', 'du', 'st', 'st.'
			}
			prefix_positions = [
				i for i, part in enumerate(parts[:-1])
				if part.lower() in last_name_prefixes
			]
			if prefix_positions:
				last_name_start = prefix_positions[0]
			name_dict['last_name'] = ' '.join(parts[last_name_start:])
			parts = parts[:last_name_start]
		for part in parts:
			nickname_match = re.match(r'[\"\'\(\[](.+)[\"\'\)\]]', part)
			if nickname_match:
				name_dict['nicknames'].append(nickname_match.group(1))
			else: name_dict['first_names'].append(part)
		return name_dict

	def parse_names_list(self, names_string):
		"""Parse a comma-separated list of names into structured records.

		This helper is designed for strings that contain multiple names in a
		compact form, such as ``"Hughes Da, Szkuta B, Biermann Tw,"``. Each
		parsed name is returned as a dictionary with normalized fields for first
		names, last name, nicknames, and suffix. For compact ``Last Initials``
		forms, all-uppercase 1-3 letter final tokens are treated as initials;
		this allows short surnames such as ``Kim`` and ``Lee`` to remain surnames.
		The name order is inferred from the list as a whole and applied
		consistently to names whose token lengths are ambiguous.

		Expected input:
		    names_string: A raw string containing one or more names separated by
		        commas, with optional suffixes and nickname markers.

		Expected output:
		    A list of dictionaries, one per parsed name. Each dictionary has the
		    same normalized structure produced by :meth:`parse_name`.
		"""
		# 1. Split on your newly added commas
		raw_names = [n.strip() for n in names_string.split(',') if n.strip()]
		parsed_results = []

		# Infer a shared order from the clearest initial patterns in the list.
		# Short-token length alone is weaker evidence because surnames can be short.
		order_scores = {'first_last': 0, 'last_first': 0}
		strong_order_scores = {'first_last': 0, 'last_first': 0}
		for raw_name in raw_names:
			parts = raw_name.split()
			if parts and parts[-1].lower().replace('.', '') in self.suffixes:
				parts = parts[:-1]
			if len(parts) < 2:
				continue

			first_token, last_token = parts[0], parts[-1]
			first_is_initial = (
				len(first_token.rstrip('.')) <= 3
				and (len(first_token.rstrip('.')) == 1 or first_token.isupper())
			)
			last_is_initial = (
				len(last_token.rstrip('.')) <= 3
				and (len(last_token.rstrip('.')) == 1 or last_token.isupper())
			)

			if first_is_initial and not last_is_initial:
				strong_order_scores['first_last'] += 1
			elif last_is_initial and not first_is_initial:
				strong_order_scores['last_first'] += 1
			elif len(first_token) > 3 and len(last_token) <= 3:
				order_scores['last_first'] += 1
			elif len(last_token) > 3 and len(first_token) <= 3:
				order_scores['first_last'] += 1

		name_order = None
		scores_to_use = (
			strong_order_scores
			if any(strong_order_scores.values())
			else order_scores
		)
		if scores_to_use['first_last'] != scores_to_use['last_first']:
			name_order = max(scores_to_use, key=scores_to_use.get)
		
		for name in raw_names:
			name = " ".join(name.split()).strip()
			if not name:
				continue
				
			name_dict = {'first_names': [], 'last_name': '', 'nicknames': [], 'suffix': ''}
			
			# 2. Extract Suffixes
			last_tok = re.split(r'[\s,]', name)[-1]
			if last_tok.lower().replace('.', '') in self.suffixes:
				name_dict['suffix'] = last_tok
				name = name[:-len(last_tok)].strip().rstrip(',')

			# 3. Explicit "Last, First" check (if any internal commas exist)
			if ',' in name:
				parts = re.split(r',', name, maxsplit=1)
				name_dict['last_name'] = parts[0].strip()
				name = parts[1].strip()

			# 4. Tokenise into name elements
			tokens = re.findall(r'(?:\w+\.+\w*|[\w\"\'\(\[\]\)-]+)', name)
			if not tokens:
				if name_dict['last_name']:
					parsed_results.append(name_dict)
				continue

			# 5. Extract Last Name (support both "First Last" and compact "Last Initials")
			if not name_dict['last_name']:
				ln_prefixes = {
					'van', 'von', 'de', 'di', 'da', 'le', 'del', 'du',
					'der', 'den', 'la', 'st.', 'st'
				}

				# Uppercase short tokens are initials. Recognizing them independently
				# of surname length supports short surnames such as "Kim J" and "Lee TW".
				tail_is_initial = re.fullmatch(r'[A-Z]{1,3}\.?', tokens[-1]) is not None
				use_last_first = name_order == 'last_first' or (
					name_order is None
					and tokens[-1].lower().rstrip('.') not in ln_prefixes
					and (
						tail_is_initial
						or (len(tokens[-1].rstrip('.')) <= 3 and len(tokens[0]) > 3)
					)
				)
				if use_last_first and (
					tokens[0].lower() in ln_prefixes or '-' in tokens[0]
				):
					split_idx = 1
					while split_idx < len(tokens) and tokens[split_idx].lower() in ln_prefixes:
						split_idx += 1
					if split_idx < len(tokens):  
						split_idx += 1
					name_dict['last_name'] = " ".join(tokens[:split_idx])
					tokens = tokens[split_idx:]
				elif use_last_first:
					name_dict['last_name'] = tokens[0]
					tokens = tokens[1:]
					
				# Standard fallback: compound surname or "First Last".
				else:
					# Keep particles immediately before the final token with the
					# surname, e.g. "M DE PAGTER" -> first name M, last name DE PAGTER.
					last_name_start = len(tokens) - 1
					while (
						last_name_start > 0
						and tokens[last_name_start - 1].lower() in ln_prefixes
					):
						last_name_start -= 1

					if last_name_start < len(tokens) - 1:
						name_dict['last_name'] = " ".join(tokens[last_name_start:])
						tokens = tokens[:last_name_start]
					elif tokens[0].lower() in ln_prefixes or '-' in tokens[0]:
						split_idx = 1
						while split_idx < len(tokens) and tokens[split_idx].lower() in ln_prefixes:
							split_idx += 1
						if split_idx < len(tokens):
							split_idx += 1
						name_dict['last_name'] = " ".join(tokens[:split_idx])
						tokens = tokens[split_idx:]
					else:
						name_dict['last_name'] = tokens[-1]
						tokens = tokens[:-1]

			# 6. Distribute remaining tokens to First Names or Nicknames
			for part in tokens:
				nickname_match = re.match(r'[\"\'\(\[](.+)[\"\'\)\]]', part)
				if nickname_match:
					name_dict['nicknames'].append(nickname_match.group(1))
				else:
					name_dict['first_names'].append(part.upper())
					
			parsed_results.append(name_dict)

		return parsed_results

	def _suffix_sim(self, suffix1, suffix2):
		"""Return a similarity score for two suffix strings.

		Equivalent suffixes such as ``Jr`` and ``Junior`` are treated as a match,
		while missing suffixes receive a partial score of ``0.5``.

		Expected input:
		    suffix1: Suffix string from the first parsed name.
		    suffix2: Suffix string from the second parsed name.

		Expected output:
		    A float between 0 and 1 representing suffix agreement. Exact matches
		    return 1, missing/partial matches return 0.5, and incompatible suffixes
		    return 0.
		"""
		s1, s2 = suffix1.strip('.').lower(), suffix2.strip('.').lower()
		if s1 == s2: return 1
		if any(s == 'junior' for s in [s1, s2]) \
		and any(s == 'jr' for s in [s1, s2]): return 1
		if any(s == 'senior' for s in [s1, s2]) \
		and any(s == 'sr' for s in [s1, s2]): return 1
		if not s1 or not s2: return 0.5
		return 0

	def _suffix_acron_letter(self, suffix):
		"""Map common suffix abbreviations to a canonical marker.

		This is used by the matching logic to recognize suffix-based acronym
		relationships without needing to compare full text strings.

		Expected input:
		    suffix: A suffix string such as ``'jr'``, ``'sr'``, or ``'ii'``.

		Expected output:
		    A canonical marker string (typically ``'j'``) when the suffix is a
		    Junior-like abbreviation; otherwise an empty string.
		"""
		if suffix and (suffix.lower()[0] == 'j' or suffix.lower() == 'ii'):
			return 'j'
		return ''

	def _max_subseq_sim(self, fnames1, fnames2, jr1='', jr2=''):
		"""Compare first-name sequences allowing for initials and acronym matches.

		This helper scans the shorter first-name list across the longer one and
		returns the best similarity found for any subsequence alignment.

		Expected input:
		    fnames1: A list of first-name tokens from the first parsed name.
		    fnames2: A list of first-name tokens from the second parsed name.
		    jr1: Optional Junior-like suffix marker for the first name.
		    jr2: Optional Junior-like suffix marker for the second name.

		Expected output:
		    A float similarity score for the best matching alignment of the two
		    first-name token sequences.
		"""
		name_order = sorted([(fnames1, jr1), (fnames2, jr2)], key=lambda x:len(x[0]))
		(shorter, jrshort), (longer, jrlong) = name_order
		if not shorter: return 0
		sequence_sims = []
		last_initial_match = False
		for s in range(len(longer) - len(shorter) + 1):
			token_sims = []
			for t in range(len(shorter)):
				if len(shorter[t])==1 and len(longer[s+t])==1 \
				and shorter[t][0]==longer[s+t][0]:
					token_sims.append(1)
					last_initial_match = True
				elif len(shorter[t])==1 or len(longer[s+t])==1:
					if shorter[t][0]==longer[s+t][0]:
						token_sims.append(self.params['disc_initial'])
						last_initial_match = True
					else:
						if last_initial_match:
							if (len(shorter[t])==1 and shorter[t][0] == jrlong) \
							or (len(longer[s+t])==1 and longer[s+t][0] == jrshort):
								token_sims.append(self.params['disc_acronym'])
						else: token_sims.append(self.params['disc_initial_mismatch'])
						last_initial_match = False
				else:
					substr_sim = self._max_substr_sim(shorter[t], longer[s+t])
					if len(shorter[t])==2 and shorter[t][0] == longer[s+t][0] \
					and ((s+t+1 < len(longer) and shorter[t][1] == longer[s+t+1][0]) \
					or shorter[t][1] == jrlong):
						if self.params['disc_acronym'] > substr_sim:
							substr_sim = self.params['disc_acronym']
					elif len(longer[s+t])==2 and longer[s+t][0] == shorter[t][0] \
					and ((t+1 < len(shorter) and longer[s+t][1] == shorter[t+1][0]) \
					or longer[s+t][1] == jrshort):
						if self.params['disc_acronym'] > substr_sim:
							substr_sim = self.params['disc_acronym']
					token_sims.append(substr_sim)
					last_initial_match = False
			if token_sims: sequence_sims.append(mean(token_sims))
		sim = max(sequence_sims)
		if sequence_sims[0] != sim: sim *= self.params['disc_missing_fname']
		elif len(shorter) != len(longer):
			if len(longer) == len(shorter) + 1 and jrshort and longer[-1] == 'j' \
			and self.params['disc_acronym'] > self.params['disc_missing_mname']:
				sim *= self.params['disc_acronym']
			else: sim *= self.params['disc_missing_mname']
		return sim

	def _max_substr_sim(self, str1, str2):
		"""Compute the best substring-alignment similarity between two tokens.

		The comparison rewards exact matches and partial matches while reducing
		scores for abbreviations that start later in the string.

		Expected input:
		    str1: A token or short string from one name component.
		    str2: A token or short string from another name component.

		Expected output:
		    A float similarity score representing the strongest aligned substring
		    match between the two inputs.
		"""
		if len(str1)==0 or len(str2)==0: return 0
		(shorter, longer) = sorted([str1, str2], key=lambda x:len(x))
		subsims = []
		for i in range(len(longer) - len(shorter) + 1):
			subsims.append(1 - self.distfun(shorter, longer[i:i+len(shorter)]))
		sim = max(subsims)
		if len(longer)!=len(shorter):
			sim *= self.params['disc_abbrev']
		if subsims.index(max(subsims)) > 0:
			sim *= self.params['disc_abbrev_notstart']
		return sim

	def _weighted_sum(self, sims_dict):
		"""Combine component similarities using the class weighting scheme.

		Expected input:
		    sims_dict: A dictionary with keys ``'first_names'``, ``'last_name'``,
		        and ``'suffix'`` whose values are similarity scores in the range
		        [0, 1].

		Expected output:
		    A single weighted similarity score computed using the class-defined
		    weights for each component.
		"""
		sims_list = [sims_dict[key] for key in self.weight_order]
		sims_weighted = np.multiply(sims_list, self.params['weights'])
		return sum(sims_weighted)

	def find_closest_names(self, target_names, other_names, min_sim=0.90):
		"""Find the closest match for each target name within a candidate list.

		This method sorts candidate names by their standardized form and then scans
		neighboring entries to locate the highest-scoring match under the
		:method:`match_names` logic.

		Expected input:
		    target_names: A name string, parsed-name dictionary, or list of either.
		        Parsed-name dictionaries must contain ``first_names``,
		        ``last_name``, ``nicknames``, and ``suffix``.
		    other_names: Candidate name strings, parsed-name dictionaries, or a
		        list of either. Parsed-name dictionaries have the same required
		        keys as above.
			min_sim: Minimum similarity threshold for a match to be returned. If no
		        candidate exceeds this threshold, ``None`` is returned for that
		        target.

		Expected output:
		    A list of parsed-name dictionaries, one per target, in target order.
		    Each item is the highest-scoring candidate from ``other_names``, or
		    ``None`` when there are no candidates.
		"""
		def normalize_names(names, argument_name):
			if isinstance(names, (str, dict)):
				names = [names]
			elif not isinstance(names, list):
				raise TypeError(
					f"{argument_name} must be a name string, parsed-name dictionary, or list"
				)

			normalized = []
			for name in names:
				if isinstance(name, str):
					normalized.append(self.parse_name(name))
				elif isinstance(name, dict):
					normalized_name = name.copy()
					last_name = normalized_name['last_name']
					first_names = list(normalized_name['first_names'])
					surname_particles = {
						'da', 'de', 'del', 'den', 'der', 'di', 'du',
						'la', 'le', 'st', 'st.', 'van', 'von',
					}
					prefix_start = len(first_names)
					while (
						prefix_start > 0
						and first_names[prefix_start - 1].lower() in surname_particles
					):
						prefix_start -= 1
					if last_name and prefix_start < len(first_names):
						last_name = ' '.join(first_names[prefix_start:] + [last_name])
						first_names = first_names[:prefix_start]
					normalized_name['last_name'] = last_name
					normalized_name['first_names'] = first_names
					normalized.append(normalized_name)
				else:
					raise TypeError(
						f"Each item in {argument_name} must be a string or parsed-name dictionary"
					)
			return normalized

		targets_dicts = normalize_names(target_names, 'target_names')
		others_dicts = normalize_names(other_names, 'other_names')
		if not others_dicts:
			return [None] * len(targets_dicts)

		others_enum = [(self._name_string_std(name_dict), name_dict, i)
						for i, name_dict in enumerate(others_dicts)]
		others_sorted = sorted(others_enum)
		matches = []
		for target_dict in targets_dicts:
			target_std = self._name_string_std(target_dict)
			i = bisect_left(others_sorted, (target_std,))
			best_sim, best_i = 0, i
			if i == len(others_sorted): best_i = i - 1
			j = i - 1
			cont_left = i > 0
			cont_right = j < len(others_sorted) - 1
			while cont_left or cont_right:
				if cont_left and i > 0:
					i -= 1
					sims = {'suffix': self._suffix_sim(target_dict['suffix'], others_sorted[i][1]['suffix']),
						'last_name': 1 - self.distfun(target_dict['last_name'], others_sorted[i][1]['last_name']),
						'first_names': 1}
					if self._weighted_sum(sims) < best_sim:
						cont_left = False
					else:
						sim = self.match_names(target_std, others_sorted[i][0], speed='slow')
						if sim > best_sim:
							best_sim, best_i = sim, i
				else:
					cont_left = False
				if cont_right and j < len(others_sorted) - 1:
					j += 1
					sims = {'suffix': self._suffix_sim(target_dict['suffix'], others_sorted[j][1]['suffix']),
						'last_name': 1 - self.distfun(target_dict['last_name'], others_sorted[j][1]['last_name']),
						'first_names': 1}
					if self._weighted_sum(sims) < best_sim:
						cont_right = False
					else:
						sim = self.match_names(target_std, others_sorted[j][0], speed='slow')
						if sim > best_sim:
							best_sim, best_i = sim, j
				else:
					cont_right = False
			if best_sim < min_sim:
				matches.append({})
			else:
				matches.append(others_sorted[best_i][1])
		return matches

	def _name_string_std(self, name_dict):
		"""Return a canonical string representation of a parsed name.

		This standardized format is used for sorting and candidate search
		operations before a full similarity comparison is performed.

		Expected input:
		    name_dict: A parsed-name dictionary returned by :meth:`parse_name`.

		Expected output:
		    A normalized string in the form ``'Last, First Names Suffix'`` with
		    surrounding whitespace removed.
		"""
		return (name_dict['last_name'] + ', ' \
				+ ' '.join(name_dict['first_names']) + ' ' \
				+ name_dict['suffix']).strip()
