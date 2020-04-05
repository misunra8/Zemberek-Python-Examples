from os.path import join
import os,re,collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
# from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_20newsgroups

from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM

## 00 Bilisim/Bilgiaayar
## 01 Cevre
## 10 Ekonomi
## 11 Edebiyat

if __name__ == '__main__':

    ZEMBEREK_PATH: str = join('bin', 'zemberek-full.jar')

    ### ZEMBEREK INIT
    startJVM(
        getDefaultJVMPath(),
        '-ea',
        f'-Djava.class.path={ZEMBEREK_PATH}',
        convertStrings=False
    )

    TurkishSpellChecker: JClass = JClass(
        'zemberek.normalization.TurkishSpellChecker'
    )
    TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
    TurkishLexer: JClass = JClass('zemberek.tokenization.antlr.TurkishLexer')
    TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
    Token: JClass = JClass('zemberek.tokenization.Token')
    WordAnalysis: JClass = JClass('zemberek.morphology.analysis.WordAnalysis')

    tokenizer: TurkishTokenizer = TurkishTokenizer.ALL
    morphology: TurkishMorphology = TurkishMorphology.createWithDefaults()
    spell_checker: TurkishSpellChecker = TurkishSpellChecker(morphology)

    Paths: JClass = JClass('java.nio.file.Paths')

    TurkishSentenceNormalizer: JClass = JClass(
        'zemberek.normalization.TurkishSentenceNormalizer'
    )
    Paths: JClass = JClass('java.nio.file.Paths')

    normalizer = TurkishSentenceNormalizer(
        TurkishMorphology.createWithDefaults(),
        Paths.get(
            join('data', 'normalization')
        ),
        Paths.get(
            join('data', 'lm', 'lm.2gram.slm')
        )
    )

    with open(
            join('data', 'bannedWords.txt'),
            'r',
            encoding='utf-8'
    ) as banfile:
        banlist = banfile.read().lower()

    ### DOCS READING



    ALLDOCS_COUNTER=[]
    ALLDOCS_POST = []

    LOAD_ALL = load_files('data/docsCategorized',encoding='utf-8')
    # print(LOAD_ALL)
    ALLDOCS_PREPROCESS = LOAD_ALL.data
    CATEGORIES = LOAD_ALL.target_names
    docCount = len(ALLDOCS_PREPROCESS)

    print(ALLDOCS_PREPROCESS)
    print(CATEGORIES)


    for i in range(docCount):
        makale = []
        makaleN = []

        x = ALLDOCS_PREPROCESS[i].lower()
        s = re.sub(r'[^\w\s]', '', x)  # regex noktalama temizleme
        s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s)  # numbers
        s = re.sub(r'\s+', ' ', s, flags=re.I)
        s = s.split(' ')
        print(s)


        ### NORMALIZATION

        #SPELLCHECKER

        # for word in s:
        #     suggestions = spell_checker.suggestForWord(JString(word))
        #
        # for i, word in enumerate(s):
        #     if not word.isnumeric():
        #         if spell_checker.suggestForWord(JString(word)):
        #             if not spell_checker.check(JString(word)):
        #                 s[i] = str(spell_checker.suggestForWord(JString(word))[0])


        for word in s:
            if word != "":
                x: str = str(normalizer.normalize(JString(word)))
                makale.append(x)

        for word in makale:
            results: WordAnalysis = morphology.analyze(JString(word))
            result = str(results)
            print(result)
            analysisResult = result.split("analysisResults=[")[1].lstrip().split(']')[0]
            if len(analysisResult) > 0:
                stemWord = analysisResult.split("[")[1].lstrip().split(':')[0]
                if not stemWord in banlist and not stemWord.isnumeric() and len(stemWord) > 1:
                    makaleN.append(stemWord.lower())

        # counter=dict(collections.Counter(makaleN).most_common())
        # ALLDOCS_COUNTER.append(counter)
        makaleFull = ' '.join(makaleN)
        ALLDOCS_POST.append(makaleFull)
        # print(makaleN)


    #
    vectorizer = CountVectorizer()
    # # vectorizer.fit(text)
    X_train_counts = vectorizer.fit(ALLDOCS_POST)
    # print(X_train_counts)
    print(vectorizer.vocabulary_.__len__())
    # vector = vectorizer.transform(list(ALLDOCS[0]))
    # # summarize encoded vector
    # print(vector.toarray().shape)
    print(ALLDOCS_POST)

    shutdownJVM()
