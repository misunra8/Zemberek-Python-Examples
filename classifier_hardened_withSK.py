from os.path import join
import os,re,collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import  numpy as np
import math
import matplotlib.pyplot as plt

from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM

## 00 Bilisim/Bilgiaayar
## 01 Cevre
## 10 Ekonomi
## 11 Edebiyat

BANTERMS =['Adj','Adv','Num','Conj','Num,Card','Num,Ord','Num,Dist','Pron,Demons','Postp,PCDat','Interj','Abbrv','Det',"Postp,PCAbl",'Noun,Abbrv','Pron,Ques','Postp,PCNom','Postp,PCGen','Pron,Quant','Pron,Pers','Adv,Prop','Adj,Prop','Pron,Reflex','Ques','Postp,PCIns','Adv,Time','Dup','Noun,Time']
BAN_OK_TERMS = ['Adj','Adv']
DAMNEDTERMS =['Num','Conj','Num,Card','Num,Ord','Num,Dist','Pron,Demons','Postp,PCDat','Interj','Abbrv','Det',"Postp,PCAbl",'Noun,Abbrv','Pron,Ques','Postp,PCNom','Postp,PCGen','Pron,Quant','Pron,Pers','Adv,Prop','Adj,Prop','Pron,Reflex','Ques','Postp,PCIns','Adv,Time','Dup','Noun,Time']
OKTERMS=['Verb','Noun','Noun,Prop']
OKOKTERMS=['Verb','Noun']

### LONGER STEM BETTER

def Two_OKTERMS(term1,term2):
    if term1[1] == 'Verb' and term2[1] == 'Verb':
        if len(term1[0]) >= len(term2[0]): ### LONGER BETTER APPROACH
            return term1[0]
        else:
            return term2[0]
    elif term1[1] == 'Noun' and term2[1] == 'Noun':
        if len(term1[0]) >= len(term2[0]): ### LONGER BETTER APPROACH
            return term1[0]
        else:
            return term2[0]
    elif term1[1] == 'Noun' and term2[1] == 'Verb':
        if len(term1[0]) <= len(term2[0])-3:
            return term1[0]
        else:
            return term2[0]
    elif term2[1] == 'Noun' and term1[1] == 'Verb':
        if len(term2[0]) <= len(term1[0])-3:
            return term2[0]
        else:
            return term1[0]
    elif (term1[1] in OKOKTERMS) and (term2[1] == 'Noun,Prop'):
        return term1[0]
    elif (term2[1] in OKOKTERMS) and (term1[1] == 'Noun,Prop'):
        return term2[0].lower()
    else:
        return term1[0].lower()

def Two_XORTERMS(term1,term2):
    if (term1[1] in BANTERMS) and (term2[1] in OKTERMS):
        return term2[0].lower()
    elif (term2[1] in BANTERMS) and (term1[1] in OKTERMS):
        return term1[0].lower()
    else:
        return "WTTFFFF"

def cleanse(preprocess_doc,alldocs,alldocsCount):
    makale = []
    makaleN = []

    x = preprocess_doc.lower()
    s = re.sub(r'[^\w\s]', '', x)  # regex noktalama temizleme
    s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s)  # numbers
    s = re.sub(r'\s+', ' ', s, flags=re.I)
    s = s.split(' ')

    ### NORMALIZATION

    # SPELLCHECKER

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

    ### GODMODE ###
    ### verb > noun > noun,prop
    for word in makale:
        results: WordAnalysis = morphology.analyze(JString(word))
        result = str(results)
        result2 = (result.split("analysisResults=[")[1].lstrip().split('}')[0]).split(", ")
        if not result2 == [']']:  ### MEANS EMPTY / NO VALID RESPONSE FROM ZEMBEREK
            stemMatrix = [[] for i in range(len(result2))]
            result3 = []
            for i in range(len(result2)):
                a = result2[i].split("[")[1].lstrip().split(':')[0]
                b = result2[i].split(":")[1].lstrip().split(']')[0]
                a = a + "---" + b
                result3.append(a)
            allUniqueStems = list(set(result3))  ### TOP PRIORITY
            for j in range(len(allUniqueStems)):
                allUniqueStems[j] = allUniqueStems[j].split("---")


            ##### GODMODE ENHANCED #####

            if len(allUniqueStems) == 1:
                if allUniqueStems[0][1] in BANTERMS:
                    pass
                elif allUniqueStems[0][1] in OKTERMS:  # in
                    makaleN.append(allUniqueStems[0][0].lower())
                else:
                    print("wtf :", allUniqueStems[0])
            #
            if len(allUniqueStems) == 2:
                if (allUniqueStems[0][1] in BANTERMS) and (allUniqueStems[1][1] in BANTERMS):
                    pass
                elif (allUniqueStems[0][1] in OKTERMS) and (allUniqueStems[1][1] in OKTERMS):  # in
                    makaleN.append(Two_OKTERMS(allUniqueStems[0], allUniqueStems[1]))
                else:  # in
                    makaleN.append(Two_XORTERMS(allUniqueStems[0], allUniqueStems[1]))

            if len(allUniqueStems) >= 3:
                badNum = 0
                goodNum = 0
                adjNum = 0
                advNum = 0
                gg = 0
                maybe = 0
                nice = 0
                for i in range(len(allUniqueStems)):
                    if allUniqueStems[i][1] in BANTERMS:
                        badNum += 1
                        if allUniqueStems[i][1] in DAMNEDTERMS:
                            gg += 1
                        if word == allUniqueStems[i][0]:
                            if allUniqueStems[i][1] in BAN_OK_TERMS:
                                maybe += 1

                    elif allUniqueStems[i][1] in OKTERMS:
                        goodNum += 1
                        if word == allUniqueStems[i][0]:
                            if allUniqueStems[i][1] in OKOKTERMS:
                                nice += 1

                    if allUniqueStems[i][1] == 'Adj':
                        adjNum += 1
                    elif allUniqueStems[i][1] == 'Adv':
                        advNum += 1

                if (badNum > 2 and goodNum >= badNum and nice >= 1):  ## kesin alinacak
                    makaleN.append(word.lower())

                elif (
                        badNum == 2 and goodNum > 0 and adjNum >= 2 and gg == 0 and goodNum > badNum - 2):  ## mutlak kurtarim

                    newStems = []

                    for i in range(len(allUniqueStems)):
                        if allUniqueStems[i][1] in OKOKTERMS:
                            newStems.append(allUniqueStems[i])

                    if len(newStems) == 1:
                        makaleN.append(newStems[0][0].lower())

                    elif len(newStems) == 2:
                        makaleN.append(Two_OKTERMS(newStems[0], newStems[1]))

                    elif len(newStems) == 3:
                        makaleN.append(newStems[0][0].lower())


                elif goodNum > badNum and gg >= 1:
                    pass

                elif badNum == 0 or badNum < goodNum:  # hepsi alinacak
                    nouns = []
                    verbs = []
                    for i in range(len(allUniqueStems)):
                        if allUniqueStems[i][1] == 'Noun':
                            nouns.append(allUniqueStems[i])
                        elif allUniqueStems[i][1] == 'Verb':
                            verbs.append(allUniqueStems[i])
                    noun = ""
                    verb = ""
                    long = ""
                    if len(nouns) >= 2:
                        temp = nouns[0]
                        for i in range(len(nouns)):
                            if len(nouns[i][0]) >= len(nouns[0][0]):
                                temp = nouns[i]
                        noun = temp
                    if len(verbs) >= 2:
                        temp = verbs[0]
                        for i in range(len(verbs)):
                            if len(verbs[i][0]) >= len(verbs[0][0]):
                                temp = verbs[i]
                        verb = temp

                    if len(nouns) == 0 and len(verbs) == 0:
                        temp = allUniqueStems[0]
                        for i in range(len(allUniqueStems)):
                            if len(allUniqueStems[i][0]) >= len(allUniqueStems[0][0]):
                                temp = allUniqueStems[i]
                        long = temp[0]

                    if len(noun) > 0 and len(verb) > 0:  # in
                        makaleN.append(Two_OKTERMS(noun, verb))
                    elif verb == "" and noun == "":  ## tek elemanli diziler OR lu
                        if len(nouns) == 1 and len(verbs) == 1:
                            makaleN.append(Two_OKTERMS(nouns[0], verbs[0]))
                        elif len(nouns) == 1 and len(verbs) == 0:
                            makaleN.append(nouns[0][0].lower())
                        elif len(nouns) == 0 and len(verbs) == 1:
                            makaleN.append(verbs[0][0].lower())
                        else:
                            makaleN.append(long.lower())
                    elif verb == "":
                        makaleN.append(noun[0].lower())
                    elif noun == "":
                        makaleN.append(verb[0].lower())

    counter = dict(collections.Counter(makaleN).most_common())
    alldocsCount.append(counter)
    makaleFull = ' '.join(makaleN)
    alldocs.append(makaleN)
    global AD_CV_SCIKIT,Vocab
    AD_CV_SCIKIT.append(makaleFull)
    Vocab += makaleN


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

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


    ### DOCS READING

    LOAD_ALL = load_files('data/docsCategorized', encoding='utf-8')

    ALLDOCS_PREPROCESS = LOAD_ALL.data
    CATEGORIES = LOAD_ALL.target_names
    docCount = len(ALLDOCS_PREPROCESS)
    # docCount = 2

    ALLDOCS_COUNTER=[]
    ALLDOCS = []
    # print(LOAD_ALL)

    AD_CV_SCIKIT = []
    Vocab = []



    for i in range(docCount):
        cleanse(ALLDOCS_PREPROCESS[i],ALLDOCS,ALLDOCS_COUNTER)

    # print(ALLDOCS_COUNTER)
    Vocab = sorted(list(dict.fromkeys(Vocab)))
    # print(Vocab.__len__())


    # WORKING COUNT VECTOR
    COUNT_VECTOR = np.empty([docCount,Vocab.__len__()],dtype=int)

    for i in range(docCount):
        temp = np.zeros(len(Vocab),dtype=int)
        for j in range(len(ALLDOCS[i])):
            if ALLDOCS[i][j] in Vocab:
                temp[Vocab.index(ALLDOCS[i][j])] = ALLDOCS_COUNTER[i].get(ALLDOCS[i][j])
        COUNT_VECTOR[i] = temp

    # print(COUNT_VECTOR[0])

    COUNT_VECT_DICT=[]
    for i in range(docCount):
        temp ={}
        for j in range(len(ALLDOCS[i])):
            if ALLDOCS[i][j] in Vocab:
                temp[Vocab.index(ALLDOCS[i][j])] = ALLDOCS_COUNTER[i].get(ALLDOCS[i][j])
        COUNT_VECT_DICT.append(temp)

    # print(COUNT_VECT_DICT)
    # print(ALLDOCS_COUNTER[0].get('olmak'))

    IDF_Vocab = []
    for i in Vocab:
        df = 0
        for j in range(docCount):
            if i in ALLDOCS[j]:
                df += 1
        idf = math.log(docCount+1/df+1) + 1
        IDF_Vocab.append(idf)

    TFIDF_VECTOR = np.empty([docCount,Vocab.__len__()],dtype=float)

    # TFIDF =[]
    # ### TF-IDF TRANSFORM
    # for i in range(docCount):
    #     temp = {}
    #     maxTermCount = COUNT_VECT_DICT[i].get(max(COUNT_VECT_DICT[i],key=COUNT_VECT_DICT[i].get))
    #     for j in COUNT_VECT_DICT[i]:
    #         TF = COUNT_VECT_DICT[i].get(j) / len(ALLDOCS[i])
    #         TF_IDF = TF * IDF_Vocab[j]
    #         temp[j] = TF_IDF
    #     TFIDF.append(temp)
    #     print(temp)

    for i in range(docCount):
        temp = np.zeros(len(Vocab),dtype=float)
        maxTermCount = max(COUNT_VECTOR[i])
        for j in range(len(COUNT_VECTOR[i])):
            TF = COUNT_VECTOR[i][j] / len(ALLDOCS[i])
            TF_IDF = TF * IDF_Vocab[j]
            temp[j] = TF_IDF
        TFIDF_VECTOR[i]= temp

    # print(TFIDF_VECTOR)

    CLASS_VECTORS=[]
    eko = np.zeros(len(Vocab), dtype=float)
    ede = np.zeros(len(Vocab), dtype=float)
    cev = np.zeros(len(Vocab), dtype=float)
    bil = np.zeros(len(Vocab), dtype=float)
    for i in range(docCount):
        if LOAD_ALL.target[i] == 3 :
            eko += TFIDF_VECTOR[i]
        elif LOAD_ALL.target[i] == 2:
            ede += TFIDF_VECTOR[i]
        elif LOAD_ALL.target[i] == 1:
            cev += TFIDF_VECTOR[i]
        elif LOAD_ALL.target[i] == 0:
            bil += TFIDF_VECTOR[i]

    eko = eko /5
    ede = ede /5
    cev = cev /5
    bil = bil /5


    ekoPos = TFIDF_VECTOR[15] - eko
    edePos = TFIDF_VECTOR[15] - ede
    cevPos = TFIDF_VECTOR[15] - cev
    bilPos = TFIDF_VECTOR[15] - bil

    print(np.sum(np.absolute(ekoPos))/4)
    print(np.sum(np.absolute(edePos))/4)
    print(np.sum(np.absolute(cevPos))/4)
    print(np.sum(np.absolute(bilPos))/4)

    print(LOAD_ALL.target)
    # fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    # axs[0, 0].plot(eko/5)
    # axs[0, 1].plot(ede/5)
    # axs[1, 0].plot(cev/5)
    # axs[1, 1].plot(bil/5)
    # plt.show()



    # # print(TFIDF)
    # cVector = CountVectorizer()
    # AllDocsCV = cVector.fit_transform(AD_CV_SCIKIT) ## CV = count vectorizer
    # print(AllDocsCV)
    # print(AllDocsCV.shape)
    #
    # tfidf_transformer = TfidfTransformer()
    # AllDocsTFIDF = tfidf_transformer.fit_transform(AllDocsCV)
    # print(AllDocsTFIDF)
    #
    # clf = MultinomialNB().fit(AllDocsTFIDF, LOAD_ALL.target)
    # print(clf)
    #
    #
    # with open(
    #         join('data','testdoc', 'test.txt'),
    #         'r',
    #         encoding='utf-8'
    # ) as testFile:
    #     test = testFile.read().lower()
    #
    # LOAD_PRE = test
    # # print(test)
    #
    #
    # LOAD_POST = []
    # cleanse(LOAD_PRE,LOAD_POST)
    #
    # rVectorX = cVector.transform(LOAD_POST)
    # loadTFIDF = TfidfTransformer().fit_transform(rVectorX)
    # predicted = clf.predict(loadTFIDF)
    #
    # ## 0 Bilisim/Bilgiaayar
    # ## 1 Cevre
    # ## 2 Edebiyat
    # ## 3 Ekonomi
    # if predicted == 0 :
    #     print("Bilisim")
    # elif predicted == 1 :
    #     print("Cevre")
    # elif predicted == 2 :
    #     print("Edebiyat")
    # elif predicted == 3:
    #     print("Ekonomi")

    shutdownJVM()
