import pos_model

def main() :
    model = pos_model.PoStagger()
    model.compile_dataset()
    model.compile_model()
    #model.train_model()

    return

if __name__ == "__main__" :
    main()