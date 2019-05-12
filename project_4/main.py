import pos_model
import metrics

def main() :
    model = pos_model.PoStagger()
    model.compile_dataset()
    model.compile_model()
    model.train_model()
    model.load_model()
    metrics.evaluate_model(model)
    model.test_model()

    print(model.predict(['Feels', 'like', 'you', 'are', 'in', 'Brooklyn', ',', 'but', 'people', 'watching', 'is', 'entertaining', '.']))
    print(model.predict(['We', 'honestly', 'can', 'not', 'think', 'of', 'even', '1', 'thing', 'we', 'did', 'n\'t', 'like', '!']))
    print(model.predict(['Bland', 'and', 'over', 'cooked', '.']))
    print(model.predict(['Great', 'meets', 'that', 'are', 'already', 'cooked', ',', 'easy', 'to', 'take', 'home', 'for', 'dinner', '.']))
    print(model.predict(['The', 'talk', 'of', 'the', 'day', 'besides', 'a', 'more', 'level', 'playing', 'field', 'with', 'China', 'was', 'North', 'Korea', '.']))
    print(model.predict(['The', 'workers', 'sped', 'up', 'and', 'down', 'the', 'street', 'with', 'no', 'mind', 'to', 'the', 'small', 'children', 'playing', '.']))
    print(model.predict(['They', 're', 'probably', 'just', 'drawn', 'for', 'the', 'show', 'anyways', '.']))
    print(model.predict(['One', 'of', 'the', 'pictures', 'shows', 'a', 'flag', 'that', 'was', 'found', 'in', 'Fallujah', '.']))
    print(model.predict([ "I", "love", "you", "."]))
    print(model.predict(["I", "am", "in", "love", "with", "the", "giant", "plate", "of", "nachos" "!"]))
    print(model.predict(["Would", "love", "for", "you", "to", "join", "us", "."]))
    print(model.predict(["Let", "'s", "make", "love", "."]))

    return

if __name__ == "__main__" :
    main()