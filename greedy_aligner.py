import numpy as np


default_delta = lambda x, y: 1 if x == y else -1


def align_global(v1, v2, delta=None, minimize=False):
    if delta is None:
        delta = default_delta
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
    def __init__(self, strings, labels=None, alphabet=None, delta=None):
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
        self._delta = default_delta if delta is None else delta
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
        profile = np.zeros((len(self.alphabet), strings.shape[1]))
        for j in range(strings.shape[1]):
            chars = list(strings[:, j])
            for i in range(len(self.alphabet)):
                profile[i, j] = chars.count(self.alphabet[i]) / strings.shape[0]
        return profile

    def _tau(self, x, j):
        res = 0
        for i, y in enumerate(self.alphabet):
            res += self.profile[i, j] * self._delta(x, y)
        return res
    
    def _sigma(self, i, q, j):
        res = 0
        for ix, x in enumerate(self.alphabet):
            for iy, y in enumerate(q.alphabet):
                res += (self.profile[ix, i] * q.profile[iy, j] *
                        self._delta(x, y))
        return res
     
    def add_str(self, v, label=None):
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
        score_mat = np.zeros((self.profile.shape[1] + 1, 
                              profile.profile.shape[1] + 1))
        case_mat = np.zeros_like(score_mat, dtype='uint8')

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
        i, j = profile.profile.shape[0], profile.profile.shape[1]
        path = []
        while i != 0 and j != 0:
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
        for i, j, case in path:
            ii, jj = i - 1, j - 1
            if case == 1:   # add gap to myself at location ii
                my_strings = np.insert(my_strings, ii, '-', axis=1)
            elif case == 2: # add gap to the other profile at location jj
                its_strings = np.insert(its_strings, jj, '-', axis=1)
        strings = np.concatenate([my_strings, its_strings])
        self._strings = strings
        self.profile = self._build_profile_from_strings(strings)
        if labels is None:
            labels = ['v%d' % (len(self._string_labels) + 1 + i)
                        for i in range(its_strings.shape[0])]
        else:
            assert len(labels) == its_strings.shape[0]
        self._string_labels += labels

        return score_mat[-1, -1]


if __name__ == '__main__':
    profile1 = Profile(['GTCTGA', 'GTCAGC'])
    # print(profile.add_str('GATATT'))
    # print(profile.add_str('GATTCA'))
    profile2 = Profile(['GATTCA', 'GATATT'])
    print(profile1)
    print(profile2)
    print(profile1.add_profile(profile2))
    print(profile1)