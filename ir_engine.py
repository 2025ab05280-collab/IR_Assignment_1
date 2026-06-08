import re
import math
import os
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure NLTK data is downloaded
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK resource {res}: {e}")

# Call resource download
download_nltk_resources()

class IREngine:

    def __init__(self):
        self.documents = {}   # doc_id -> raw_text
        self.doc_names = {}   # doc_id -> doc_name
        self.stop_words = set()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback stop words if NLTK fails
            self.stop_words = {
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent',
                'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
                'cannot', 'could', 'did', 'do', 'does', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
                'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my',
                'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
                'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their',
                'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
                'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
                'whom', 'why', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
            }

        self.stemmer = PorterStemmer()
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

        # Indexes
        self.inverted_index_stem = defaultdict(list)
        self.inverted_index_lem = defaultdict(list)
        self.biword_index = defaultdict(list)
        self.positional_index = defaultdict(lambda: defaultdict(list))
        self.k_gram_index = defaultdict(set)

        # Vocabulary sets
        self.vocab_stem = set()
        self.vocab_lem = set()
        self.vocab_raw = set()

    def add_document(self, doc_id, name, text):
        self.documents[doc_id] = text
        self.doc_names[doc_id] = name

    def clear_documents(self):
        self.documents.clear()
        self.doc_names.clear()
        self.inverted_index_stem.clear()
        self.inverted_index_lem.clear()
        self.biword_index.clear()
        self.positional_index.clear()
        self.k_gram_index.clear()
        self.vocab_stem.clear()
        self.vocab_lem.clear()
        self.vocab_raw.clear()

    # Preprocessing Steps

    def preprocess_pipeline(self, text):
        """
        Runs the full step-by-step pipeline and returns a dictionary of intermediate outputs.
        """
        steps = {}
        steps['original'] = text

        # 1. Lowercasing
        lowercased = text.lower()
        steps['lowercase'] = lowercased

        # 2. Hyphen Handling
        hyphen_handled = re.sub(r'-', ' ', lowercased)
        steps['hyphen_handled'] = hyphen_handled

        # 3. Tokenization
        tokens = re.findall(r'\b\w+\b', hyphen_handled)
        steps['tokenized'] = list(tokens)

        # 4. Stop word removal
        no_stopwords = [t for t in tokens if t not in self.stop_words]
        steps['stop_words_removed'] = no_stopwords

        # 5. Stemming (Porter)
        stemmed = [self.stemmer.stem(t) for t in no_stopwords]
        steps['stemmed'] = stemmed

        # 6. Lemmatization (WordNet)
        if self.lemmatizer:
            try:
                lemmatized = [self.lemmatizer.lemmatize(t) for t in no_stopwords]
            except Exception:
                lemmatized = list(no_stopwords)
        else:
            lemmatized = list(no_stopwords)
        steps['lemmatized'] = lemmatized

        return steps

    def build_indexes(self):
        """
        Builds Inverted (Stemmed & Lemmatized), Biword, Positional, and K-gram indexes.
        """
        self.inverted_index_stem.clear()
        self.inverted_index_lem.clear()
        self.biword_index.clear()
        self.positional_index.clear()
        self.k_gram_index.clear()
        self.vocab_stem.clear()
        self.vocab_lem.clear()
        self.vocab_raw.clear()

        for doc_id, text in self.documents.items():
            pipeline = self.preprocess_pipeline(text)

            raw_tokens = pipeline['tokenized']
            no_stopwords = pipeline['stop_words_removed']
            stemmed_tokens = pipeline['stemmed']
            lemmatized_tokens = pipeline['lemmatized']

            # 1. Stemmed Inverted Index & Vocab
            for term in stemmed_tokens:
                self.vocab_stem.add(term)
                if doc_id not in self.inverted_index_stem[term]:
                    self.inverted_index_stem[term].append(doc_id)

            # 2. Lemmatized Inverted Index & Vocab
            for term in lemmatized_tokens:
                self.vocab_lem.add(term)
                if doc_id not in self.inverted_index_lem[term]:
                    self.inverted_index_lem[term].append(doc_id)

            # 3. Raw Vocab for tolerant retrieval / spell correction
            for term in no_stopwords:
                self.vocab_raw.add(term)

            # 4. Biword Index
            for i in range(len(no_stopwords) - 1):
                biword = f"{no_stopwords[i]}_{no_stopwords[i+1]}"
                if doc_id not in self.biword_index[biword]:
                    self.biword_index[biword].append(doc_id)

            # 5. Positional Index
            for pos, term in enumerate(raw_tokens):
                self.positional_index[term][doc_id].append(pos)

        # 6. Build K-gram index (k=2 and k=3) from raw vocabulary
        for term in self.vocab_raw:
            self._add_to_k_gram(term)

    def _add_to_k_gram(self, term):
        padded = f"^{term}$"
        # 2-grams
        for i in range(len(padded) - 1):
            gram = padded[i:i+2]
            self.k_gram_index[gram].add(term)
        # 3-grams
        for i in range(len(padded) - 2):
            gram = padded[i:i+3]
            self.k_gram_index[gram].add(term)

    # ──────────────────────────────────────────────────────────────────────────
    # Phrase Query Processing
    # ──────────────────────────────────────────────────────────────────────────

    def search_biword_phrase(self, phrase):
        """
        Splits the phrase into biwords and intersects their postings.
        Returns matching doc_ids and details about execution.
        """
        phrase = phrase.lower()
        phrase_tokens = re.findall(r'\b\w+\b', phrase)
        filtered_tokens = [t for t in phrase_tokens if t not in self.stop_words]

        if len(filtered_tokens) == 0:
            return [], []

        if len(filtered_tokens) == 1:
            term = filtered_tokens[0]
            matches = []
            for doc_id, text in self.documents.items():
                if term in text.lower():
                    matches.append(doc_id)
            return sorted(matches), [f"Single term query: '{term}' - performed standard document scan."]

        biwords = []
        for i in range(len(filtered_tokens) - 1):
            biwords.append(f"{filtered_tokens[i]}_{filtered_tokens[i+1]}")

        result_docs = None
        logs = []
        logs.append(f"Query split into biwords: {biwords}")

        for bw in biwords:
            postings = self.biword_index.get(bw, [])
            logs.append(f"Postings list for biword '{bw}': {postings}")
            if result_docs is None:
                result_docs = set(postings)
            else:
                result_docs = result_docs.intersection(postings)

        res = sorted(list(result_docs)) if result_docs is not None else []
        logs.append(f"Final Biword intersection results: {res}")
        return res, logs

    def search_positional_phrase(self, phrase):
        """
        Performs positional intersection for multi-word phrase query.
        """
        phrase = phrase.lower()
        phrase_tokens = re.findall(r'\b\w+\b', phrase)

        if len(phrase_tokens) == 0:
            return [], []

        logs = []
        logs.append(f"Query parsed into positional tokens: {phrase_tokens}")

        postings_lists = []
        for term in phrase_tokens:
            postings = self.positional_index.get(term, {})
            postings_lists.append(postings)
            logs.append(f"Postings for '{term}': {dict(postings)}")

        common_docs = None
        for plist in postings_lists:
            docs = set(plist.keys())
            if common_docs is None:
                common_docs = docs
            else:
                common_docs = common_docs.intersection(docs)

        if not common_docs:
            logs.append("No documents contain all terms of the phrase.")
            return [], logs

        logs.append(f"Documents containing all terms: {list(common_docs)}")

        matching_docs = []
        for doc_id in common_docs:
            pos_lists = [plist[doc_id] for plist in postings_lists]
            if self._check_consecutive(pos_lists):
                matching_docs.append(doc_id)
                logs.append(f"Doc {doc_id}: Positional match found!")
            else:
                logs.append(f"Doc {doc_id}: Terms present but NOT consecutive.")

        return sorted(matching_docs), logs

    def _check_consecutive(self, pos_lists):
        """
        Helper to check if there exists consecutive positions in lists.
        pos_lists is a list of lists of integers: [[pos_t1...], [pos_t2...], ...]
        """
        if not pos_lists:
            return False

        candidates = set(pos_lists[0])

        for offset, next_positions in enumerate(pos_lists[1:], start=1):
            next_pos_set = set(next_positions)
            candidates = {c for c in candidates if (c + offset) in next_pos_set}
            if not candidates:
                return False

        return len(candidates) > 0

    # ──────────────────────────────────────────────────────────────────────────
    # Tolerant Retrieval (Wildcards and Spelling)
    # ──────────────────────────────────────────────────────────────────────────

    def search_wildcard(self, wildcard_query):
        """
        Searches using K-gram index and intersects, then filters with regex.
        """
        wildcard_query = wildcard_query.lower()

        escaped = re.escape(wildcard_query)
        regex_pattern = "^" + escaped.replace(r'\*', '.*') + "$"
        regex = re.compile(regex_pattern)

        segments = wildcard_query.split('*')
        k_grams = []

        for idx, seg in enumerate(segments):
            if not seg:
                continue

            padded_seg = seg
            if idx == 0:
                padded_seg = "^" + seg
            if idx == len(segments) - 1:
                padded_seg = seg + "$"

            if len(seg) >= 2:
                for i in range(len(padded_seg) - 1):
                    k_grams.append(padded_seg[i:i+2])
            if len(seg) >= 3:
                for i in range(len(padded_seg) - 2):
                    k_grams.append(padded_seg[i:i+3])

        matching_terms = None
        logs = [f"Wildcard regex pattern: {regex_pattern}"]
        logs.append(f"Extracted k-grams: {k_grams}")

        for gram in k_grams:
            terms = self.k_gram_index.get(gram, set())
            if matching_terms is None:
                matching_terms = set(terms)
            else:
                matching_terms = matching_terms.intersection(terms)

        if matching_terms is None:
            matching_terms = self.vocab_raw

        final_terms = [t for t in matching_terms if regex.match(t)]
        logs.append(f"Filtered terms: {final_terms}")

        matching_docs = set()
        for term in final_terms:
            lemmed = self.lemmatizer.lemmatize(term) if self.lemmatizer else term
            docs = self.inverted_index_lem.get(lemmed, [])
            matching_docs.update(docs)

        return sorted(list(matching_docs)), final_terms, logs

    def edit_distance(self, str1, str2):
        """
        Levenshtein Edit Distance algorithm.
        """
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )

        return dp[m][n]

    def get_spelling_suggestions(self, misspelled_word, max_suggestions=3, max_dist=2):
        """
        Finds candidate terms in the dictionary within edit distance.
        """
        misspelled_word = misspelled_word.lower()
        candidates = []

        for term in self.vocab_raw:
            dist = self.edit_distance(misspelled_word, term)
            if dist <= max_dist:
                candidates.append((term, dist))

        candidates.sort(key=lambda x: (x[1], x[0]))
        return [c[0] for c in candidates[:max_suggestions]]

    # ──────────────────────────────────────────────────────────────────────────
    # Stemming vs Lemmatization Evaluation Metrics
    # ──────────────────────────────────────────────────────────────────────────

    def compute_tfidf_similarity(self, query, doc_id, use_stemming=True):
        """
        Computes cosine similarity of TF-IDF vectors between a query and a document.
        """
        pipeline = self.preprocess_pipeline(query)
        query_terms = pipeline['stemmed'] if use_stemming else pipeline['lemmatized']

        if not query_terms:
            return 0.0

        index = self.inverted_index_stem if use_stemming else self.inverted_index_lem
        vocab = self.vocab_stem if use_stemming else self.vocab_lem
        N = len(self.documents)

        if N == 0:
            return 0.0

        idf = {}
        for term in vocab:
            df = len(index.get(term, []))
            idf[term] = math.log(N / df) if df > 0 else 0.0

        doc_text = self.documents.get(doc_id, "")
        doc_pipeline = self.preprocess_pipeline(doc_text)
        doc_terms = doc_pipeline['stemmed'] if use_stemming else doc_pipeline['lemmatized']

        doc_tf = defaultdict(int)
        for term in doc_terms:
            doc_tf[term] += 1

        doc_vector = {}
        for term, tf in doc_tf.items():
            if term in idf:
                doc_vector[term] = tf * idf[term]

        query_tf = defaultdict(int)
        for term in query_terms:
            query_tf[term] += 1

        query_vector = {}
        for term, tf in query_tf.items():
            if term in idf:
                query_vector[term] = tf * idf[term]

        dot_product = 0.0
        for term in query_vector:
            if term in doc_vector:
                dot_product += query_vector[term] * doc_vector[term]

        doc_norm = math.sqrt(sum(v**2 for v in doc_vector.values()))
        query_norm = math.sqrt(sum(v**2 for v in query_vector.values()))

        if doc_norm == 0.0 or query_norm == 0.0:
            return 0.0

        return dot_product / (doc_norm * query_norm)

    # ──────────────────────────────────────────────────────────────────────────
    # Phonetic Correction (Soundex Algorithm)
    # ──────────────────────────────────────────────────────────────────────────

    def soundex(self, word):
        """
        Computes the Soundex phonetic code for a given word.
        """
        if not word:
            return ""

        word = word.upper()
        first_letter = word[0]

        soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2',
            'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }

        encoded = first_letter
        prev_code = soundex_map.get(first_letter, '0')

        for char in word[1:]:
            code = soundex_map.get(char, '0')
            if code == '0':
                prev_code = '0'
                continue
            if code != prev_code:
                encoded += code
                prev_code = code

        encoded = encoded[:4].ljust(4, '0')
        return encoded

    def get_phonetic_suggestions(self, query_word, max_suggestions=10):
        """
        Finds vocabulary terms that share the same Soundex phonetic code
        as the query word.

        Returns:
            (matches, query_code) where
              matches    : list of matching vocabulary terms (sorted A-Z)
              query_code : the Soundex code computed for query_word
        """
        query_word = query_word.lower().strip()
        query_code = self.soundex(query_word)
        matches = []

        for term in self.vocab_raw:
            if term == query_word:
                continue
            term_code = self.soundex(term)
            if term_code == query_code:
                matches.append(term)

        matches = sorted(matches)[:max_suggestions]
        return matches, query_code

    def get_phonetic_index(self):
        """
        Builds a full Soundex phonetic index over the entire raw vocabulary.
        Returns:
            dict { soundex_code (str) -> [term1, term2, ...] }
        """
        phonetic_index = defaultdict(list)
        for term in self.vocab_raw:
            code = self.soundex(term)
            phonetic_index[code].append(term)

        return {code: sorted(terms) for code, terms in phonetic_index.items()}
