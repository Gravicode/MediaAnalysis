using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using Version = Mosaik.Core.Version;
using P = Catalyst.PatternUnitPrototype;
namespace MediaAnalysisWeb.Helpers
{
    public class CatalystEngine
    {
        public CatalystEngine()
        {
            //Initialize the English built-in models
            Catalyst.Models.Indonesian.Register();
            Catalyst.Models.English.Register();
        }
        #region entity recognition
        public async Task<string> RecognizeEntities(string Body, Language lang= Language.English)
        {
            
                // For training an AveragePerceptronModel, check the source-code here: https://github.com/curiosity-ai/catalyst/blob/master/Catalyst.Training/src/TrainWikiNER.cs
                // This example uses the pre-trained WikiNER model, trained on the data provided by the paper "Learning multilingual named entity recognition from Wikipedia", Artificial Intelligence 194 (DOI: 10.1016/j.artint.2012.03.006)
                // The training data was sourced from the following repository: https://github.com/dice-group/FOX/tree/master/input/Wikiner

                //Configures the model storage to use the online repository backed by the local folder ./catalyst-models/

                //Create a new pipeline for the english language, and add the WikiNER model to it
                Console.WriteLine("Loading models... This might take a bit longer the first time you run this sample, as the models have to be downloaded from the online repository");
                var nlp = await Pipeline.ForAsync(lang);
                nlp.Add(await AveragePerceptronEntityRecognizer.FromStoreAsync(language: lang, version: Mosaik.Core.Version.Latest, tag: "WikiNER"));

                //Another available model for NER is the PatternSpotter, which is the conceptual equivalent of a RegEx on raw text, but operating on the tokenized form off the text.
                //Adds a custom pattern spotter for the pattern: single("is" / VERB) + multiple(NOUN/AUX/PROPN/AUX/DET/ADJ)
                var isApattern = new PatternSpotter(lang, 0, tag: "is-a-pattern", captureTag: "IsA");
                isApattern.NewPattern(
                    "Is+Noun",
                    mp => mp.Add(
                        new PatternUnit(P.Single().WithToken("is").WithPOS(PartOfSpeech.VERB)),
                        new PatternUnit(P.Multiple().WithPOS(PartOfSpeech.NOUN, PartOfSpeech.PROPN, PartOfSpeech.AUX, PartOfSpeech.DET, PartOfSpeech.ADJ))
                ));
                nlp.Add(isApattern);

                //For processing a single document, you can call nlp.ProcessSingle
                var doc = new Document(Body, lang);
                nlp.ProcessSingle(doc);

                return GetEntities(doc);
                /*
                //For correcting Entity Recognition mistakes, you can use the Neuralyzer class. 
                //This class uses the Pattern Matching entity recognition class to perform "forget-entity" and "add-entity" 
                //passes on the document, after it has been processed by all other proceses in the NLP pipeline
                var neuralizer = new Neuralyzer(lang, 0, "WikiNER-sample-fixes");

                //Teach the Neuralyzer class to forget the match for a single token "Amazon" with entity type "Location"
                neuralizer.TeachForgetPattern("Location", "Amazon", mp => mp.Add(new PatternUnit(P.Single().WithToken("Amazon").WithEntityType("Location"))));

                //Teach the Neuralyzer class to add the entity type Organization for a match for the single token "Amazon"
                neuralizer.TeachAddPattern("Organization", "Amazon", mp => mp.Add(new PatternUnit(P.Single().WithToken("Amazon"))));

                //Add the Neuralyzer to the pipeline
                nlp.UseNeuralyzer(neuralizer);

                //Now you can see that "Amazon" is correctly recognized as the entity type "Organization"
                var doc2 = new Document(Data.Sample_1, lang);
                nlp.ProcessSingle(doc2);
                PrintDocumentEntities(doc2);
                */
            
        }

        private string RecognizeEntitiesWithSpotter(string Body, SpotterInfo info,Mosaik.Core.Language lang=Language.English)
        {
            //Another way to perform entity recognition is to use a gazeteer-like model. For example, here is one for capturing a set of programing languages
            var spotter = new Spotter(Language.Any, 0, info.Tag, info.CaptureTag);
            spotter.Data.IgnoreCase = true; //In some cases, it might be better to set it to false, and only add upper/lower-case exceptions as required
            foreach(var entry in info.Entries)
            {
                spotter.AddEntry(entry);
            }
            var nlp = Pipeline.TokenizerFor(lang);
            nlp.Add(spotter); //When adding a spotter model, the model propagates any exceptions on tokenization to the pipeline's tokenizer

            var doc = new Document(Body, lang);

            nlp.ProcessSingle(doc);

            return GetEntities(doc);
        }
        private static string GetEntities(IDocument doc)
        {
            var result = ($"Input:\n\t'{doc.Value}'\n\nTokenized Value:\n\t'{doc.TokenizedValue(mergeEntities: true)}'\n\nEntities: \n{string.Join("\n", doc.SelectMany(span => span.GetEntities()).Select(e => $"\t{e.Value} [{e.EntityType.Type}]"))}");
            return result;
        }
        #endregion

        #region Lang Detector

        public enum LangDetector
        {
            CLD2,FastText,Unknown
        }
        public async Task< Language> DetectLanguage(string Body, LangDetector Mode)
        {
          
            //This example shows the two language detection models available on Catalyst. 
            //The first is derived from the Chrome former language detection code Compact Language Detector 2 (https://github.com/CLD2Owners/cld2)
            //and the newer model is derived from Facebook's FastText language detection dataset (see: https://fasttext.cc/blog/2017/10/02/blog-post.html)

            //Configures the model storage to use the local folder ./catalyst-models/
            Storage.Current = new DiskStorage("catalyst-models");

            switch (Mode)
            {
                case LangDetector.CLD2:
                    var cld2LanguageDetector = await LanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");
                    var doc2 = new Document(Body);
                    cld2LanguageDetector.Process(doc2);
                    return doc2.Language;
                    break;
                case LangDetector.FastText:
                    var fastTextLanguageDetector = await FastTextLanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");
                    var doc = new Document(Body);
                    fastTextLanguageDetector.Process(doc);
                    return doc.Language;
                    break;
            }
            return Language.Unknown;
            /*
            // You can also access all predictions via the Predict method:
            var allPredictions = fastTextLanguageDetector.Predict(new Document(Data.LongSamples[Language.Spanish]));

            Console.WriteLine($"\n\nTop 10 predictions and scores for the Spanish sample:");
            foreach (var kv in allPredictions.OrderByDescending(kv => kv.Value).Take(10))
            {
                Console.WriteLine($"{kv.Key.ToString().PadRight(40)}\tScore: {kv.Value:n2}");
            }*/
        }
        #endregion

        #region LDA
        public async Task<bool> TrainLDA(List<string> TrainingDataSet, List<string> TestDataSet,string Tag="my-lda", Language lang = Language.English,int NumTopics=20)
        {
            try
            {
                //Configures the model storage to use the local folder ./catalyst-models/
                Storage.Current = new DiskStorage("catalyst-models");

                //Download the Reuters corpus if necessary
                //var (train, test) = await Corpus.Reuters.GetAsync();
                var train = new List<IDocument>();
                foreach (var item in TrainingDataSet)
                {
                    train.Add(new Document(item, lang));
                }
                var test = new List<IDocument>();
                foreach (var item in TestDataSet)
                {
                    test.Add(new Document(item, lang));
                }
                //Parse the documents using the English pipeline, as the text data is untokenized so far
                var nlp = Pipeline.For(lang);

                var trainDocs = nlp.Process(train).ToArray();
                var testDocs = nlp.Process(test).ToArray();


                //Train an LDA topic model on the trainind dateset
                using (var lda = new LDA(Language.English, 0, Tag))
                {
                    lda.Data.NumberOfTopics = NumTopics; //Arbitrary number of topics
                    lda.Train(trainDocs, Environment.ProcessorCount);
                    await lda.StoreAsync();
                }
                using (var lda = await LDA.FromStoreAsync(Language.English, 0, Tag))
                {
                    foreach (var doc in testDocs)
                    {
                        if (lda.TryPredict(doc, out var topics))
                        {
                            var docTopics = string.Join("\n", topics.Select(t => lda.TryDescribeTopic(t.TopicID, out var td) ? $"[{t.Score:n3}] => {td.ToString()}" : ""));

                            Console.WriteLine("------------------------------------------");
                            Console.WriteLine(doc.Value);
                            Console.WriteLine("------------------------------------------");
                            Console.WriteLine(docTopics);
                            Console.WriteLine("------------------------------------------\n\n");
                        }
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return false;
            }
            
        }

        public async Task<List<(string topic, double score)>> PredictLDA(string Body, string Tag = "my-lda", Language lang = Language.English)
        {
            //Configures the model storage to use the local folder ./catalyst-models/
            Storage.Current = new DiskStorage("catalyst-models");


            IDocument test = new Document(Body, lang);

            //Parse the documents using the English pipeline, as the text data is untokenized so far
            var nlp = Pipeline.For(lang);

            var testDoc = nlp.ProcessSingle(test);
            using (var lda = await LDA.FromStoreAsync(Language.English, 0, Tag))
            {

                if (lda.TryPredict(testDoc, out var topics))
                {
                    List<(string topic, double score)> result = new();
                    topics.ToList().ForEach(t =>
                    {
                        var res = lda.TryDescribeTopic(t.TopicID, out var td);
                        if (res)
                            result.Add((td.ToString(), t.Score));
                    });
                    return result;
                    /*
                    Console.WriteLine("------------------------------------------");
                    Console.WriteLine(testDoc.Value);
                    Console.WriteLine("------------------------------------------");
                    Console.WriteLine(docTopics);
                    Console.WriteLine("------------------------------------------\n\n");
                    */
                }

            }
            return default;
        }
        #endregion
    }

    public class SpotterInfo
    {
        public string Tag { get; set; }
        public string CaptureTag { get; set; }
        public List<string> Entries { get; set; }
    }
}
