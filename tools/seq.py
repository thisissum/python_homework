def pad_sequence(ids, padding=0, length=None):
    """pad sequences to target length
    If length is None, pad to max length of sequences
    args:
        ids: List[List[int]] or List[int], the sequences of word ids
        padding: int, the num used to represent <pad> signal
        length: int, the target length to pad to, if None: max length will be used
    returns:
        ids: sequences of ids after padding
    """
    if isinstance(ids[0], int):

        assert length is not None, "To pad a single sentence, you must specify length"
        if len(ids) < length:
            ids += [padding for _ in range(length - len(ids))]
        elif len(ids) > length:
            ids = ids[:length]
        else:
            pass

    if isinstance(ids[0], list):
        
        if length is None:
            length = max(map(lambda x:len(x), ids))
            
        else:
            for i, line in enumerate(ids):
                if len(line) > length:
                    ids[i] = line[:length]
                elif len(line) < length:
                    dif = length - len(line) 
                    ids[i] = line + dif * [padding]
                else:
                    pass

    return ids


def filt_stopword(seqs, stopword_list):
    """Given many lists of word sequence, filt the stopword in stopword_list
    args:
        seqs: List[List[str]], the sequences of words
        stopword_list: List[str], a list of stopword
    returns:
        filted_seqs: List[List[str]], the input that filted by stopword list
    """
    filt_sentence = lambda sentence: list(filter(lambda x:x not in stopword_list, sentence))
    filted_seqs = list(map(filt_sentence, seqs))
    return filted_seqs


def select_str_by_bool(string, bool_seq, level='char'):
    """select a string by the given bool sequence in order
    args:
        string: str, the string to select
        bool_seq: list of bool or array of bool, must be same length with string
        mode: str, 'char' or 'word', the level to select, 
              if 'char', return a list of single char, if 'word', connected True will be connected together
    example:

    """
    assert len(string) == len(bool_seq), "input string and bool_seq must have same length"
    char_bool_pair = zip(string, bool_seq)
    output = []
    for i, (char, condition) in enumerate(char_bool_pair):
        if condition:
            output.append(char)
            if level == 'word':
                if i+1 < len(bool_seq):
                    if bool_seq[i+1] == False:
                        output.append('|||||||')
                else:
                    pass
    if level == 'word':
        if output[-7:] == '|||||||':
            output = output[:-7]
        output = ''.join(output).split('|||||||')
    else:
        pass
    return output