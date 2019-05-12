import pos_model

def main() :
    model = pos_model.PoStagger()
    #model.compile_dataset()
    model.compile_model()
    model.train_model()

    model.load_model()

    '''
    this PRON
    is AUX
    a DET
    terrorist ADJ
    organization NOUN
    plain ADJ
    and CCONJ
    simple ADJ
    . PUNCT
    '''

    #This is a terrorist organization plain and simple.
    #print(model.predict(['This', 'is', 'a', 'terrorist', 'organization', 'plain', 'and', 'simple', '.' ]))
    model.test_model()
    return

if __name__ == "__main__" :
    main()