import numpy as np


def default_delta_func(x, y):
    """
    Default delta function. If there two characters are the same, 
    give a score of 1; otherwise (including the case of a gap), give
    a a score of -1.

    Alternative delta functions should be defined similarly.
    
    Attributes
    ----------
    x, y : char
        The characters in the alphabet (including `-` for a gap).
    
    Returns
    -------
    float
        The delta score.
    
    """
    if x == y:
        return 1.0
    else:
        return -1.0


def align_global(v1, v2, delta=default_delta_func, minimize=False):
    """
    Apply global sequence-to-sequence alignment using the Needleman–
    Wunsch algorithm.
    
    Parameters
    ----------
    v1, v2 : str
        The two input sequences.
    delta : function, optional
        The character-to-character score function. The default value 
        is ``default_delta_func``.
    minimize : bool, optional
        If True, the aligner will minimize, instead of maximize, the 
        final score. It should be set to True for the classical edit 
        distance problem. The default value is False.

    Returns
    -------
    str
        The aligned sequence ``v1``, with gaps denoted by ``-``.
    str
        The aligned sequence ``v2``, with gaps denoted by ``-``.
    float
        The final alignment score.

    Example
    -------
        >>> align_global('ACTTGAC', 'ACGTGC')
        ('ACTTGAC', 'ACGT-GC', 3.0)
    
    """
    score_mat = np.zeros((len(v1) + 1, len(v2) + 1))
    case_mat = np.zeros_like(score_mat, dtype='uint8')
    selector = np.argmin if minimize else np.argmax

    # Table filling
    for i in range(1, len(v1) + 1):
        for j in range(1, len(v2) + 1):
            ii, jj = i - 1, j - 1
            scores = [
                score_mat[i - 1, j] + delta(v1[ii], '-'),
                score_mat[i, j - 1] + delta('-', v2[jj]),
                score_mat[i - 1, j - 1] + delta(v1[ii], v2[jj])
            ]
            argmax = selector(scores)
            score_mat[i, j] = scores[argmax]
            case_mat[i, j] = argmax + 1

    # Backtrace
    i, j = len(v1), len(v2)
    path = []
    while i != 0 and j != 0:
        case = case_mat[i, j]
        path.append((i, j, case))
        if case == 1:   # from top (add gap to v2)
            i -= 1
        elif case == 2: # from left (add gap to v1)
            j -= 1
        else:   # align
            i -= 1
            j -= 1
    path.reverse()

    # Update strings
    v1_padded, v2_padded = list(v1), list(v2)
    for i, j, case in path:
        ii, jj = i - 1, j - 1
        if case == 1:   # add gap to v2 at location jj
            v2_padded.insert(jj, '-')
        elif case == 2: # add gap to v1 at location ii
            v1_padded.insert(ii, '-')
    final_len = max(len(v1_padded), len(v2_padded))
    if len(v1_padded) < final_len:
        v1_padded = v1_padded + ['-'] * (final_len - len(v1_padded))
    if len(v2_padded) < final_len:
        v2_padded = v2_padded + ['-'] * (final_len - len(v2_padded))
    
    return ''.join(v1_padded), ''.join(v2_padded), score_mat[-1, -1]
    


class Profile:
    """
    A multisequence profile representing the frequencies of character 
    appearances at each position.

    Attributes
    ----------
    alphabet : list
        A list of allowed characters, including the gap character.
    profile : ndarray
        The 2-dimensional profile table.

    Methods
    -------
    add_profile(profile, labels=None)
        Add a profile to the current Profile object and perform 
        alignment.
    add_str(v, label=None)
        Add a sequence to the current Profile object and perform
        alignment.
    copy()
        Return a copy of the current Profile object.
    
    """
    def __init__(self, strings, labels=None, alphabet=None, 
                 delta=default_delta_func):
        """
        Initialize a profile.
        
        Parameters
        ----------
        strings : list
            A list of strings each denoting a sequence. They must have
            identical lengths.
        labels : list, optional
            The string labels of the sequences. If not specified, 
            labels of the format ``v<x>`` will be assigned where
            ``<x>`` is an integer.
        alphabet : list, optional
            A list of allowed characters, including the gap character.
            If not specified, it is assumed that all characters are 
            covered in the input sequences.
        delta : function, optional
            The character-to-character score function. The default value 
            is ``default_delta_func``.

        Example
        -------
        >>> profile = Profile(['ACTTGAC', 'ACGTGCC', 'ACGGCAC'],
        ...                   labels=['seq1', 'seq2', 'seq3'],
        ...                   alphabet=['A', 'C', 'T', 'G', '-'])
        <__main__.Profile at 0x7ff8dbec3898>

        """
        assert len(strings) > 0
        for s in strings:
            assert len(s) == len(strings[0])
        if alphabet is None:
            alphabet = sorted(list({x for x in s for s in [*strings, '-']}))
        self.alphabet = alphabet
        if labels is None:
            labels = ['v%d' % (i + 1) for i in range(len(strings))]
        else:
            assert len(labels) == len(strings)
        self._string_labels = labels
        self._delta = delta
        self._strings = np.array([list(s) for s in strings])
        self.profile = self._build_profile_from_strings(self._strings)
    
    def __str__(self):
        lines = []
        # Print strings
        lines.append('Alignment:')
        for i, s in enumerate(self._strings):
            lines.append(self._string_labels[i] + '\t' + '\t'.join(s))
        lines.append('')
        # Print profile
        lines.append('Profile:')
        for i in range(self.profile.shape[0]):
            s = self.alphabet[i] + '\t'
            s += '\t'.join(['%.2f' % x for x in self.profile[i, :]])
            lines.append(s)
        return '\n'.join(lines)
    
    def copy(self):
        res = Profile.__new__(Profile)
        res.alphabet = self.alphabet[:]
        res._string_labels = self._string_labels[:]
        res._delta = self._delta
        res._strings = self._strings.copy()
        res.profile = self.profile.copy()
        return res

    def _build_profile_from_strings(self, strings):
        """
        Build profile table from a list of strings of equal lengths.
        """
        profile = np.zeros((len(self.alphabet), strings.shape[1]))
        for j in range(strings.shape[1]):
            chars = list(strings[:, j])
            for i in range(len(self.alphabet)):
                profile[i, j] = chars.count(self.alphabet[i]) / strings.shape[0]
        return profile

    def _tau(self, x, j):
        """
        Calculate the :math:`\tau` score of character ``x`` and column 
        ``j``.
        """
        res = 0
        for i, y in enumerate(self.alphabet):
            res += self.profile[i, j] * self._delta(x, y)
        return res
    
    def _sigma(self, i, q, j):
        """
        Calculate the :math:`\sigma` score between the ith column of
        the current profile and the jth column of the profile ``q``.
        """
        res = 0
        for ix, x in enumerate(self.alphabet):
            for iy, y in enumerate(q.alphabet):
                res += (self.profile[ix, i] * q.profile[iy, j] *
                        self._delta(x, y))
        return res
     
    def add_str(self, v, label=None):
        """
        Add a sequence to the current Profile object and perform
        alignment.

        Parameters
        ----------
        v : str
            The new sequence to add.
        label : str, optional
            The label of the new sequence. If not specified, ``v<x>``
            will be used where ``<x>`` is an integer.

        Returns
        -------
        float
            The alignment score.

        """
        score_mat = np.zeros((len(v) + 1, self.profile.shape[1] + 1))
        case_mat = np.zeros_like(score_mat, dtype='uint8')
        
        # Table filling
        for i in range(1, len(v) + 1):
            for j in range(1, self.profile.shape[1] + 1):
                ii, jj = i - 1, j - 1
                scores = [
                    score_mat[i - 1, j] + self._delta(v[ii], '-'),
                    score_mat[i, j - 1] + self._tau('-', jj),
                    score_mat[i - 1, j - 1] + self._tau(v[ii], jj)
                ]
                argmax = np.argmax(scores)
                score_mat[i, j] = scores[argmax]
                case_mat[i, j] = argmax + 1
        
        # Backtrace
        i, j = len(v), self.profile.shape[1]
        path = []
        while i != 0 and j != 0:
            case = case_mat[i, j]
            path.append((i, j, case))
            if case == 1:   # from top (add gap to profile)
                i -= 1
            elif case == 2: # from left (add gap to string)
                j -= 1
            else:   # align
                i -= 1
                j -= 1
        path.reverse()

        # Update profile
        strings = self._strings
        v_padded = np.array(list(v)).reshape(1, -1)
        for i, j, case in path:
            ii, jj = i - 1, j - 1
            if case == 1:   # add gap to profile at location jj
                strings = np.insert(strings, jj, '-', axis=1)
            elif case == 2: # add gap to string at location ii
                v_padded = np.insert(v_padded, ii, '-', axis=1)
        strings = np.concatenate([strings, v_padded])
        self._strings = strings
        self.profile = self._build_profile_from_strings(strings)
        if label is None:
            label = 'v%d' % (len(self._string_labels) + 1)
        self._string_labels.append(label)

        return score_mat[-1, -1]

    def add_profile(self, profile, labels=None):
        """
        Add a profile to the current Profile object and perform 
        alignment.

        Parameters
        ----------
        profile : object:Profile
            The profile to add to the current profile.
        label : list, optional
            The labels of sequences in the new profile. If not 
            specified, ``v<x>`` will be used where ``<x>`` is an 
            integer.

        Returns
        -------
        float
            The alignment score.

        """
        score_mat = np.zeros((self.profile.shape[1] + 1, 
                              profile.profile.shape[1] + 1))
        case_mat = np.zeros_like(score_mat, dtype='uint8')
        my_num_cols = self.profile.shape[1]
        its_num_cols = profile.profile.shape[1]

        # Table filling
        for i in range(1, score_mat.shape[0]):
            for j in range(1, score_mat.shape[1]):
                ii, jj = i - 1, j - 1
                scores = [
                    score_mat[i - 1, j] + self._tau('-', ii),
                    score_mat[i, j - 1] + profile._tau('-', jj),
                    score_mat[i - 1, j - 1] + self._sigma(ii, profile, jj)
                ]
                argmax = np.argmax(scores)
                score_mat[i, j] = scores[argmax]
                case_mat[i, j] = argmax + 1
        
        # Backtrace
        i, j = self.profile.shape[1], profile.profile.shape[1]
        path = []
        while i != 0 or j != 0:
            if i == 0:      # add more gaps to the other profile
                case = 2
            elif j == 0:    # add more gaps to myself
                case = 1
            else:
                case = case_mat[i, j]
            path.append((i, j, case))
            if case == 1:   # from top (add gap to myself)
                i -= 1
            elif case == 2: # from left (add gap to the other profile)
                j -= 1
            else:   # align
                i -= 1
                j -= 1
        path.reverse()

        # Update profile
        my_strings = self._strings
        its_strings = profile._strings
        m, n = my_strings.shape[0], its_strings.shape[0]
        final_strings = np.empty((m + n, len(path)), dtype='<U1')
        final_strings[:, :] = '-'
        for c, (i, j, case) in enumerate(path):
            if case == 1:
                final_strings[:m, c] = my_strings[:, i - 1]
            elif case == 2:
                final_strings[m:, c] = its_strings[:,  j - 1]
            elif case == 3:
                final_strings[:m, c] = my_strings[:, i - 1]
                final_strings[m:, c] = its_strings[:,  j - 1]
        self._strings = final_strings
        self.profile = self._build_profile_from_strings(final_strings)
        if labels is None:
            labels = ['v%d' % (len(self._string_labels) + 1 + i)
                        for i in range(its_strings.shape[0])]
        else:
            assert len(labels) == its_strings.shape[0]
        self._string_labels += labels

        return score_mat[-1, -1]


def multi_align(sequences, labels=None, 
                delta=default_delta_func, alphabet='ATCG-'):
    if labels is None:
        labels = ['v%d' % (i + 1) for i in range(len(sequences))]
    profiles = [Profile([seq], labels=[label], alphabet=alphabet, delta=delta)
                    for seq, label in zip(sequences, labels)]
    while len(profiles) > 1:
        new_profiles = []
        for i in range(0, len(profiles), 2):
            # handle the last one of an odd number of profiles
            if i + 1 == len(profiles):
                new_profiles[-1].add_profile(profiles[i])
            else:
                p1, p2 = profiles[i : i+2]
                p1.add_profile(p2)
                new_profiles.append(p1)
        profiles = new_profiles
    return profiles[0]

