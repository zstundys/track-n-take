var $s=Object.defineProperty;var Rs=(z,N,X)=>N in z?$s(z,N,{enumerable:!0,configurable:!0,writable:!0,value:X}):z[N]=X;var b=(z,N,X)=>Rs(z,typeof N!="symbol"?N+"":N,X);(function(){"use strict";const z={"adapter-transformers":["question-answering","text-classification","token-classification"],allennlp:["question-answering"],asteroid:["audio-to-audio"],bertopic:["text-classification"],diffusers:["image-to-image","text-to-image"],doctr:["object-detection"],espnet:["text-to-speech","automatic-speech-recognition"],fairseq:["text-to-speech","audio-to-audio"],fastai:["image-classification"],fasttext:["feature-extraction","text-classification"],flair:["token-classification"],k2:["automatic-speech-recognition"],keras:["image-classification"],nemo:["automatic-speech-recognition"],open_clip:["zero-shot-classification","zero-shot-image-classification"],paddlenlp:["fill-mask","summarization","zero-shot-classification"],peft:["text-generation"],"pyannote-audio":["automatic-speech-recognition"],"sentence-transformers":["feature-extraction","sentence-similarity"],setfit:["text-classification"],sklearn:["tabular-classification","tabular-regression","text-classification"],spacy:["token-classification","text-classification","sentence-similarity"],"span-marker":["token-classification"],speechbrain:["audio-classification","audio-to-audio","automatic-speech-recognition","text-to-speech","text2text-generation"],stanza:["token-classification"],timm:["image-classification","image-feature-extraction"],transformers:["audio-classification","automatic-speech-recognition","depth-estimation","document-question-answering","feature-extraction","fill-mask","image-classification","image-feature-extraction","image-segmentation","image-to-image","image-to-text","image-text-to-text","mask-generation","object-detection","question-answering","summarization","table-question-answering","text2text-generation","text-classification","text-generation","text-to-audio","text-to-speech","token-classification","translation","video-classification","visual-question-answering","zero-shot-classification","zero-shot-image-classification","zero-shot-object-detection"],mindspore:["image-classification"]},N={"text-classification":{name:"Text Classification",subtasks:[{type:"acceptability-classification",name:"Acceptability Classification"},{type:"entity-linking-classification",name:"Entity Linking Classification"},{type:"fact-checking",name:"Fact Checking"},{type:"intent-classification",name:"Intent Classification"},{type:"language-identification",name:"Language Identification"},{type:"multi-class-classification",name:"Multi Class Classification"},{type:"multi-label-classification",name:"Multi Label Classification"},{type:"multi-input-text-classification",name:"Multi-input Text Classification"},{type:"natural-language-inference",name:"Natural Language Inference"},{type:"semantic-similarity-classification",name:"Semantic Similarity Classification"},{type:"sentiment-classification",name:"Sentiment Classification"},{type:"topic-classification",name:"Topic Classification"},{type:"semantic-similarity-scoring",name:"Semantic Similarity Scoring"},{type:"sentiment-scoring",name:"Sentiment Scoring"},{type:"sentiment-analysis",name:"Sentiment Analysis"},{type:"hate-speech-detection",name:"Hate Speech Detection"},{type:"text-scoring",name:"Text Scoring"}],modality:"nlp",color:"orange"},"token-classification":{name:"Token Classification",subtasks:[{type:"named-entity-recognition",name:"Named Entity Recognition"},{type:"part-of-speech",name:"Part of Speech"},{type:"parsing",name:"Parsing"},{type:"lemmatization",name:"Lemmatization"},{type:"word-sense-disambiguation",name:"Word Sense Disambiguation"},{type:"coreference-resolution",name:"Coreference-resolution"}],modality:"nlp",color:"blue"},"table-question-answering":{name:"Table Question Answering",modality:"nlp",color:"green"},"question-answering":{name:"Question Answering",subtasks:[{type:"extractive-qa",name:"Extractive QA"},{type:"open-domain-qa",name:"Open Domain QA"},{type:"closed-domain-qa",name:"Closed Domain QA"}],modality:"nlp",color:"blue"},"zero-shot-classification":{name:"Zero-Shot Classification",modality:"nlp",color:"yellow"},translation:{name:"Translation",modality:"nlp",color:"green"},summarization:{name:"Summarization",subtasks:[{type:"news-articles-summarization",name:"News Articles Summarization"},{type:"news-articles-headline-generation",name:"News Articles Headline Generation"}],modality:"nlp",color:"indigo"},"feature-extraction":{name:"Feature Extraction",modality:"nlp",color:"red"},"text-generation":{name:"Text Generation",subtasks:[{type:"dialogue-modeling",name:"Dialogue Modeling"},{type:"dialogue-generation",name:"Dialogue Generation"},{type:"conversational",name:"Conversational"},{type:"language-modeling",name:"Language Modeling"}],modality:"nlp",color:"indigo"},"text2text-generation":{name:"Text2Text Generation",subtasks:[{type:"text-simplification",name:"Text simplification"},{type:"explanation-generation",name:"Explanation Generation"},{type:"abstractive-qa",name:"Abstractive QA"},{type:"open-domain-abstractive-qa",name:"Open Domain Abstractive QA"},{type:"closed-domain-qa",name:"Closed Domain QA"},{type:"open-book-qa",name:"Open Book QA"},{type:"closed-book-qa",name:"Closed Book QA"}],modality:"nlp",color:"indigo"},"fill-mask":{name:"Fill-Mask",subtasks:[{type:"slot-filling",name:"Slot Filling"},{type:"masked-language-modeling",name:"Masked Language Modeling"}],modality:"nlp",color:"red"},"sentence-similarity":{name:"Sentence Similarity",modality:"nlp",color:"yellow"},"text-to-speech":{name:"Text-to-Speech",modality:"audio",color:"yellow"},"text-to-audio":{name:"Text-to-Audio",modality:"audio",color:"yellow"},"automatic-speech-recognition":{name:"Automatic Speech Recognition",modality:"audio",color:"yellow"},"audio-to-audio":{name:"Audio-to-Audio",modality:"audio",color:"blue"},"audio-classification":{name:"Audio Classification",subtasks:[{type:"keyword-spotting",name:"Keyword Spotting"},{type:"speaker-identification",name:"Speaker Identification"},{type:"audio-intent-classification",name:"Audio Intent Classification"},{type:"audio-emotion-recognition",name:"Audio Emotion Recognition"},{type:"audio-language-identification",name:"Audio Language Identification"}],modality:"audio",color:"green"},"audio-text-to-text":{name:"Audio-Text-to-Text",modality:"multimodal",color:"red",hideInDatasets:!0},"voice-activity-detection":{name:"Voice Activity Detection",modality:"audio",color:"red"},"depth-estimation":{name:"Depth Estimation",modality:"cv",color:"yellow"},"image-classification":{name:"Image Classification",subtasks:[{type:"multi-label-image-classification",name:"Multi Label Image Classification"},{type:"multi-class-image-classification",name:"Multi Class Image Classification"}],modality:"cv",color:"blue"},"object-detection":{name:"Object Detection",subtasks:[{type:"face-detection",name:"Face Detection"},{type:"vehicle-detection",name:"Vehicle Detection"}],modality:"cv",color:"yellow"},"image-segmentation":{name:"Image Segmentation",subtasks:[{type:"instance-segmentation",name:"Instance Segmentation"},{type:"semantic-segmentation",name:"Semantic Segmentation"},{type:"panoptic-segmentation",name:"Panoptic Segmentation"}],modality:"cv",color:"green"},"text-to-image":{name:"Text-to-Image",modality:"cv",color:"yellow"},"image-to-text":{name:"Image-to-Text",subtasks:[{type:"image-captioning",name:"Image Captioning"}],modality:"cv",color:"red"},"image-to-image":{name:"Image-to-Image",subtasks:[{type:"image-inpainting",name:"Image Inpainting"},{type:"image-colorization",name:"Image Colorization"},{type:"super-resolution",name:"Super Resolution"}],modality:"cv",color:"indigo"},"image-to-video":{name:"Image-to-Video",modality:"cv",color:"indigo"},"unconditional-image-generation":{name:"Unconditional Image Generation",modality:"cv",color:"green"},"video-classification":{name:"Video Classification",modality:"cv",color:"blue"},"reinforcement-learning":{name:"Reinforcement Learning",modality:"rl",color:"red"},robotics:{name:"Robotics",modality:"rl",subtasks:[{type:"grasping",name:"Grasping"},{type:"task-planning",name:"Task Planning"}],color:"blue"},"tabular-classification":{name:"Tabular Classification",modality:"tabular",subtasks:[{type:"tabular-multi-class-classification",name:"Tabular Multi Class Classification"},{type:"tabular-multi-label-classification",name:"Tabular Multi Label Classification"}],color:"blue"},"tabular-regression":{name:"Tabular Regression",modality:"tabular",subtasks:[{type:"tabular-single-column-regression",name:"Tabular Single Column Regression"}],color:"blue"},"tabular-to-text":{name:"Tabular to Text",modality:"tabular",subtasks:[{type:"rdf-to-text",name:"RDF to text"}],color:"blue",hideInModels:!0},"table-to-text":{name:"Table to Text",modality:"nlp",color:"blue",hideInModels:!0},"multiple-choice":{name:"Multiple Choice",subtasks:[{type:"multiple-choice-qa",name:"Multiple Choice QA"},{type:"multiple-choice-coreference-resolution",name:"Multiple Choice Coreference Resolution"}],modality:"nlp",color:"blue",hideInModels:!0},"text-ranking":{name:"Text Ranking",modality:"nlp",color:"red"},"text-retrieval":{name:"Text Retrieval",subtasks:[{type:"document-retrieval",name:"Document Retrieval"},{type:"utterance-retrieval",name:"Utterance Retrieval"},{type:"entity-linking-retrieval",name:"Entity Linking Retrieval"},{type:"fact-checking-retrieval",name:"Fact Checking Retrieval"}],modality:"nlp",color:"indigo",hideInModels:!0},"time-series-forecasting":{name:"Time Series Forecasting",modality:"tabular",subtasks:[{type:"univariate-time-series-forecasting",name:"Univariate Time Series Forecasting"},{type:"multivariate-time-series-forecasting",name:"Multivariate Time Series Forecasting"}],color:"blue"},"text-to-video":{name:"Text-to-Video",modality:"cv",color:"green"},"image-text-to-text":{name:"Image-Text-to-Text",modality:"multimodal",color:"red",hideInDatasets:!0},"visual-question-answering":{name:"Visual Question Answering",subtasks:[{type:"visual-question-answering",name:"Visual Question Answering"}],modality:"multimodal",color:"red"},"document-question-answering":{name:"Document Question Answering",subtasks:[{type:"document-question-answering",name:"Document Question Answering"}],modality:"multimodal",color:"blue",hideInDatasets:!0},"zero-shot-image-classification":{name:"Zero-Shot Image Classification",modality:"cv",color:"yellow"},"graph-ml":{name:"Graph Machine Learning",modality:"other",color:"green"},"mask-generation":{name:"Mask Generation",modality:"cv",color:"indigo"},"zero-shot-object-detection":{name:"Zero-Shot Object Detection",modality:"cv",color:"yellow"},"text-to-3d":{name:"Text-to-3D",modality:"cv",color:"yellow"},"image-to-3d":{name:"Image-to-3D",modality:"cv",color:"green"},"image-feature-extraction":{name:"Image Feature Extraction",modality:"cv",color:"indigo"},"video-text-to-text":{name:"Video-Text-to-Text",modality:"multimodal",color:"blue",hideInDatasets:!1},"keypoint-detection":{name:"Keypoint Detection",subtasks:[{type:"pose-estimation",name:"Pose Estimation"}],modality:"cv",color:"red",hideInDatasets:!0},"visual-document-retrieval":{name:"Visual Document Retrieval",modality:"multimodal",color:"yellow",hideInDatasets:!0},"any-to-any":{name:"Any-to-Any",modality:"multimodal",color:"yellow",hideInDatasets:!0},other:{name:"Other",modality:"other",color:"blue",hideInModels:!0,hideInDatasets:!0}},X=Object.keys(N);Object.values(N).flatMap(e=>"subtasks"in e?e.subtasks:[]).map(e=>e.type),new Set(X);const wt={datasets:[{description:"A benchmark of 10 different audio tasks.",id:"s3prl/superb"},{description:"A dataset of YouTube clips and their sound categories.",id:"agkphysics/AudioSet"}],demo:{inputs:[{filename:"audio.wav",type:"audio"}],outputs:[{data:[{label:"Up",score:.2},{label:"Down",score:.8}],type:"chart"}]},metrics:[{description:"",id:"accuracy"},{description:"",id:"recall"},{description:"",id:"precision"},{description:"",id:"f1"}],models:[{description:"An easy-to-use model for command recognition.",id:"speechbrain/google_speech_command_xvector"},{description:"An emotion recognition model.",id:"ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"},{description:"A language identification model.",id:"facebook/mms-lid-126"}],spaces:[{description:"An application that can classify music into different genre.",id:"kurianbenoy/audioclassification"}],summary:"Audio classification is the task of assigning a label or class to a given audio. It can be used for recognizing which command a user is giving or the emotion of a statement, as well as identifying a speaker.",widgetModels:["MIT/ast-finetuned-audioset-10-10-0.4593"],youtubeId:"KWwzcmG98Ds"},vt={datasets:[{description:"512-element X-vector embeddings of speakers from CMU ARCTIC dataset.",id:"Matthijs/cmu-arctic-xvectors"}],demo:{inputs:[{filename:"input.wav",type:"audio"}],outputs:[{filename:"label-0.wav",type:"audio"},{filename:"label-1.wav",type:"audio"}]},metrics:[{description:"The Signal-to-Noise ratio is the relationship between the target signal level and the background noise level. It is calculated as the logarithm of the target signal divided by the background noise, in decibels.",id:"snri"},{description:"The Signal-to-Distortion ratio is the relationship between the target signal and the sum of noise, interference, and artifact errors",id:"sdri"}],models:[{description:"A speech enhancement model.",id:"ResembleAI/resemble-enhance"},{description:"A model that can change the voice in a speech recording.",id:"microsoft/speecht5_vc"}],spaces:[{description:"An application for speech separation.",id:"younver/speechbrain-speech-separation"},{description:"An application for audio style transfer.",id:"nakas/audio-diffusion_style_transfer"}],summary:"Audio-to-Audio is a family of tasks in which the input is an audio and the output is one or multiple generated audios. Some example tasks are speech enhancement and source separation.",widgetModels:["speechbrain/sepformer-wham"],youtubeId:"iohj7nCCYoM"},_t={datasets:[{description:"31,175 hours of multilingual audio-text dataset in 108 languages.",id:"mozilla-foundation/common_voice_17_0"},{description:"Multilingual and diverse audio dataset with 101k hours of audio.",id:"amphion/Emilia-Dataset"},{description:"A dataset with 44.6k hours of English speaker data and 6k hours of other language speakers.",id:"parler-tts/mls_eng"},{description:"A multilingual audio dataset with 370K hours of audio.",id:"espnet/yodas"}],demo:{inputs:[{filename:"input.flac",type:"audio"}],outputs:[{label:"Transcript",content:"Going along slushy country roads and speaking to damp audiences in...",type:"text"}]},metrics:[{description:"",id:"wer"},{description:"",id:"cer"}],models:[{description:"A powerful ASR model by OpenAI.",id:"openai/whisper-large-v3"},{description:"A good generic speech model by MetaAI for fine-tuning.",id:"facebook/w2v-bert-2.0"},{description:"An end-to-end model that performs ASR and Speech Translation by MetaAI.",id:"facebook/seamless-m4t-v2-large"},{description:"A powerful multilingual ASR and Speech Translation model by Nvidia.",id:"nvidia/canary-1b"},{description:"Powerful speaker diarization model.",id:"pyannote/speaker-diarization-3.1"}],spaces:[{description:"A powerful general-purpose speech recognition application.",id:"hf-audio/whisper-large-v3"},{description:"Latest ASR model from Useful Sensors.",id:"mrfakename/Moonshinex"},{description:"A high quality speech and text translation model by Meta.",id:"facebook/seamless_m4t"},{description:"A powerful multilingual ASR and Speech Translation model by Nvidia",id:"nvidia/canary-1b"}],summary:"Automatic Speech Recognition (ASR), also known as Speech to Text (STT), is the task of transcribing a given audio to text. It has many applications, such as voice user interfaces.",widgetModels:["openai/whisper-large-v3"],youtubeId:"TksaY_FDgnk"},kt={datasets:[{description:"Largest document understanding dataset.",id:"HuggingFaceM4/Docmatix"},{description:"Dataset from the 2020 DocVQA challenge. The documents are taken from the UCSF Industry Documents Library.",id:"eliolio/docvqa"}],demo:{inputs:[{label:"Question",content:"What is the idea behind the consumer relations efficiency team?",type:"text"},{filename:"document-question-answering-input.png",type:"img"}],outputs:[{label:"Answer",content:"Balance cost efficiency with quality customer service",type:"text"}]},metrics:[{description:"The evaluation metric for the DocVQA challenge is the Average Normalized Levenshtein Similarity (ANLS). This metric is flexible to character regognition errors and compares the predicted answer with the ground truth answer.",id:"anls"},{description:"Exact Match is a metric based on the strict character match of the predicted answer and the right answer. For answers predicted correctly, the Exact Match will be 1. Even if only one character is different, Exact Match will be 0",id:"exact-match"}],models:[{description:"A robust document question answering model.",id:"impira/layoutlm-document-qa"},{description:"A document question answering model specialized in invoices.",id:"impira/layoutlm-invoices"},{description:"A special model for OCR-free document question answering.",id:"microsoft/udop-large"},{description:"A powerful model for document question answering.",id:"google/pix2struct-docvqa-large"}],spaces:[{description:"A robust document question answering application.",id:"impira/docquery"},{description:"An application that can answer questions from invoices.",id:"impira/invoices"},{description:"An application to compare different document question answering models.",id:"merve/compare_docvqa_models"}],summary:"Document Question Answering (also known as Document Visual Question Answering) is the task of answering questions on document images. Document question answering models take a (document, question) pair as input and return an answer in natural language. Models usually rely on multi-modal features, combining text, position of words (bounding-boxes) and image.",widgetModels:["impira/layoutlm-invoices"],youtubeId:""},xt={datasets:[{description:"Wikipedia dataset containing cleaned articles of all languages. Can be used to train `feature-extraction` models.",id:"wikipedia"}],demo:{inputs:[{label:"Input",content:"India, officially the Republic of India, is a country in South Asia.",type:"text"}],outputs:[{table:[["Dimension 1","Dimension 2","Dimension 3"],["2.583383083343506","2.757075071334839","0.9023529887199402"],["8.29393482208252","1.1071064472198486","2.03399395942688"],["-0.7754912972450256","-1.647324562072754","-0.6113331913948059"],["0.07087723910808563","1.5942802429199219","1.4610432386398315"]],type:"tabular"}]},metrics:[],models:[{description:"A powerful feature extraction model for natural language processing tasks.",id:"thenlper/gte-large"},{description:"A strong feature extraction model for retrieval.",id:"Alibaba-NLP/gte-Qwen1.5-7B-instruct"}],spaces:[{description:"A leaderboard to rank text feature extraction models based on a benchmark.",id:"mteb/leaderboard"},{description:"A leaderboard to rank best feature extraction models based on human feedback.",id:"mteb/arena"}],summary:"Feature extraction is the task of extracting features learnt in a model.",widgetModels:["facebook/bart-base"]},At={datasets:[{description:"A common dataset that is used to train models for many languages.",id:"wikipedia"},{description:"A large English dataset with text crawled from the web.",id:"c4"}],demo:{inputs:[{label:"Input",content:"The <mask> barked at me",type:"text"}],outputs:[{type:"chart",data:[{label:"wolf",score:.487},{label:"dog",score:.061},{label:"cat",score:.058},{label:"fox",score:.047},{label:"squirrel",score:.025}]}]},metrics:[{description:"Cross Entropy is a metric that calculates the difference between two probability distributions. Each probability distribution is the distribution of predicted words",id:"cross_entropy"},{description:"Perplexity is the exponential of the cross-entropy loss. It evaluates the probabilities assigned to the next word by the model. Lower perplexity indicates better performance",id:"perplexity"}],models:[{description:"State-of-the-art masked language model.",id:"answerdotai/ModernBERT-large"},{description:"A multilingual model trained on 100 languages.",id:"FacebookAI/xlm-roberta-base"}],spaces:[],summary:"Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks. These models are useful when we want to get a statistical understanding of the language in which the model is trained in.",widgetModels:["distilroberta-base"],youtubeId:"mqElG5QJWUg"},St={datasets:[{description:"Benchmark dataset used for image classification with images that belong to 100 classes.",id:"cifar100"},{description:"Dataset consisting of images of garments.",id:"fashion_mnist"}],demo:{inputs:[{filename:"image-classification-input.jpeg",type:"img"}],outputs:[{type:"chart",data:[{label:"Egyptian cat",score:.514},{label:"Tabby cat",score:.193},{label:"Tiger cat",score:.068}]}]},metrics:[{description:"",id:"accuracy"},{description:"",id:"recall"},{description:"",id:"precision"},{description:"",id:"f1"}],models:[{description:"A strong image classification model.",id:"google/vit-base-patch16-224"},{description:"A robust image classification model.",id:"facebook/deit-base-distilled-patch16-224"},{description:"A strong image classification model.",id:"facebook/convnext-large-224"}],spaces:[{description:"A leaderboard to evaluate different image classification models.",id:"timm/leaderboard"}],summary:"Image classification is the task of assigning a label or class to an entire image. Images are expected to have only one class for each image. Image classification models take an image as input and return a prediction about which class the image belongs to.",widgetModels:["google/vit-base-patch16-224"],youtubeId:"tjAIM7BOYhw"},It={datasets:[{description:"ImageNet-1K is a image classification dataset in which images are used to train image-feature-extraction models.",id:"imagenet-1k"}],demo:{inputs:[{filename:"mask-generation-input.png",type:"img"}],outputs:[{table:[["Dimension 1","Dimension 2","Dimension 3"],["0.21236686408519745","1.0919708013534546","0.8512550592422485"],["0.809657871723175","-0.18544459342956543","-0.7851548194885254"],["1.3103108406066895","-0.2479034662246704","-0.9107287526130676"],["1.8536205291748047","-0.36419737339019775","0.09717650711536407"]],type:"tabular"}]},metrics:[],models:[{description:"A powerful image feature extraction model.",id:"timm/vit_large_patch14_dinov2.lvd142m"},{description:"A strong image feature extraction model.",id:"nvidia/MambaVision-T-1K"},{description:"A robust image feature extraction model.",id:"facebook/dino-vitb16"},{description:"Cutting-edge image feature extraction model.",id:"apple/aimv2-large-patch14-336-distilled"},{description:"Strong image feature extraction model that can be used on images and documents.",id:"OpenGVLab/InternViT-6B-448px-V1-2"}],spaces:[{description:"A leaderboard to evaluate different image-feature-extraction models on classification performances",id:"timm/leaderboard"}],summary:"Image feature extraction is the task of extracting features learnt in a computer vision model.",widgetModels:[]},Et={datasets:[{description:"Synthetic dataset, for image relighting",id:"VIDIT"},{description:"Multiple images of celebrities, used for facial expression translation",id:"huggan/CelebA-faces"},{description:"12M image-caption pairs.",id:"Spawning/PD12M"}],demo:{inputs:[{filename:"image-to-image-input.jpeg",type:"img"}],outputs:[{filename:"image-to-image-output.png",type:"img"}]},isPlaceholder:!1,metrics:[{description:"Peak Signal to Noise Ratio (PSNR) is an approximation of the human perception, considering the ratio of the absolute intensity with respect to the variations. Measured in dB, a high value indicates a high fidelity.",id:"PSNR"},{description:"Structural Similarity Index (SSIM) is a perceptual metric which compares the luminance, contrast and structure of two images. The values of SSIM range between -1 and 1, and higher values indicate closer resemblance to the original image.",id:"SSIM"},{description:"Inception Score (IS) is an analysis of the labels predicted by an image classification model when presented with a sample of the generated images.",id:"IS"}],models:[{description:"An image-to-image model to improve image resolution.",id:"fal/AuraSR-v2"},{description:"A model that increases the resolution of an image.",id:"keras-io/super-resolution"},{description:"A model for applying edits to images through image controls.",id:"Yuanshi/OminiControl"},{description:"A model that generates images based on segments in the input image and the text prompt.",id:"mfidabel/controlnet-segment-anything"},{description:"Strong model for inpainting and outpainting.",id:"black-forest-labs/FLUX.1-Fill-dev"},{description:"Strong model for image editing using depth maps.",id:"black-forest-labs/FLUX.1-Depth-dev-lora"}],spaces:[{description:"Image enhancer application for low light.",id:"keras-io/low-light-image-enhancement"},{description:"Style transfer application.",id:"keras-io/neural-style-transfer"},{description:"An application that generates images based on segment control.",id:"mfidabel/controlnet-segment-anything"},{description:"Image generation application that takes image control and text prompt.",id:"hysts/ControlNet"},{description:"Colorize any image using this app.",id:"ioclab/brightness-controlnet"},{description:"Edit images with instructions.",id:"timbrooks/instruct-pix2pix"}],summary:"Image-to-image is the task of transforming an input image through a variety of possible manipulations and enhancements, such as super-resolution, image inpainting, colorization, and more.",widgetModels:["stabilityai/stable-diffusion-2-inpainting"],youtubeId:""},Tt={datasets:[{description:"Dataset from 12M image-text of Reddit",id:"red_caps"},{description:"Dataset from 3.3M images of Google",id:"datasets/conceptual_captions"}],demo:{inputs:[{filename:"savanna.jpg",type:"img"}],outputs:[{label:"Detailed description",content:"a herd of giraffes and zebras grazing in a field",type:"text"}]},metrics:[],models:[{description:"A robust image captioning model.",id:"Salesforce/blip2-opt-2.7b"},{description:"A powerful and accurate image-to-text model that can also localize concepts in images.",id:"microsoft/kosmos-2-patch14-224"},{description:"A strong optical character recognition model.",id:"facebook/nougat-base"},{description:"A powerful model that lets you have a conversation with the image.",id:"llava-hf/llava-1.5-7b-hf"}],spaces:[{description:"An application that compares various image captioning models.",id:"nielsr/comparing-captioning-models"},{description:"A robust image captioning application.",id:"flax-community/image-captioning"},{description:"An application that transcribes handwritings into text.",id:"nielsr/TrOCR-handwritten"},{description:"An application that can caption images and answer questions about a given image.",id:"Salesforce/BLIP"},{description:"An application that can caption images and answer questions with a conversational agent.",id:"Salesforce/BLIP2"},{description:"An image captioning application that demonstrates the effect of noise on captions.",id:"johko/capdec-image-captioning"}],summary:"Image to text models output a text from a given image. Image captioning or optical character recognition can be considered as the most common applications of image to text.",widgetModels:["Salesforce/blip-image-captioning-large"],youtubeId:""},Ct={datasets:[{description:"Instructions composed of image and text.",id:"liuhaotian/LLaVA-Instruct-150K"},{description:"Collection of image-text pairs on scientific topics.",id:"DAMO-NLP-SG/multimodal_textbook"},{description:"A collection of datasets made for model fine-tuning.",id:"HuggingFaceM4/the_cauldron"},{description:"Screenshots of websites with their HTML/CSS codes.",id:"HuggingFaceM4/WebSight"}],demo:{inputs:[{filename:"image-text-to-text-input.png",type:"img"},{label:"Text Prompt",content:"Describe the position of the bee in detail.",type:"text"}],outputs:[{label:"Answer",content:"The bee is sitting on a pink flower, surrounded by other flowers. The bee is positioned in the center of the flower, with its head and front legs sticking out.",type:"text"}]},metrics:[],models:[{description:"Small and efficient yet powerful vision language model.",id:"HuggingFaceTB/SmolVLM-Instruct"},{description:"A screenshot understanding model used to control computers.",id:"microsoft/OmniParser-v2.0"},{description:"Cutting-edge vision language model.",id:"allenai/Molmo-7B-D-0924"},{description:"Small yet powerful model.",id:"vikhyatk/moondream2"},{description:"Strong image-text-to-text model.",id:"Qwen/Qwen2.5-VL-7B-Instruct"},{description:"Image-text-to-text model with agentic capabilities.",id:"microsoft/Magma-8B"},{description:"Strong image-text-to-text model focused on documents.",id:"allenai/olmOCR-7B-0225-preview"},{description:"Small yet strong image-text-to-text model.",id:"ibm-granite/granite-vision-3.2-2b"}],spaces:[{description:"Leaderboard to evaluate vision language models.",id:"opencompass/open_vlm_leaderboard"},{description:"Vision language models arena, where models are ranked by votes of users.",id:"WildVision/vision-arena"},{description:"Powerful vision-language model assistant.",id:"akhaliq/Molmo-7B-D-0924"},{description:"Powerful vision language assistant that can understand multiple images.",id:"HuggingFaceTB/SmolVLM2"},{description:"An application for chatting with an image-text-to-text model.",id:"GanymedeNil/Qwen2-VL-7B"},{description:"An application that parses screenshots into actions.",id:"showlab/ShowUI"},{description:"An application that detects gaze.",id:"moondream/gaze-demo"}],summary:"Image-text-to-text models take in an image and text prompt and output text. These models are also called vision-language models, or VLMs. The difference from image-to-text models is that these models take an additional text input, not restricting the model to certain use cases like image captioning, and may also be trained to accept a conversation as input.",widgetModels:["Qwen/Qwen2-VL-7B-Instruct"],youtubeId:"IoGaGfU1CIg"},Ut={datasets:[{description:"Scene segmentation dataset.",id:"scene_parse_150"}],demo:{inputs:[{filename:"image-segmentation-input.jpeg",type:"img"}],outputs:[{filename:"image-segmentation-output.png",type:"img"}]},metrics:[{description:"Average Precision (AP) is the Area Under the PR Curve (AUC-PR). It is calculated for each semantic class separately",id:"Average Precision"},{description:"Mean Average Precision (mAP) is the overall average of the AP values",id:"Mean Average Precision"},{description:"Intersection over Union (IoU) is the overlap of segmentation masks. Mean IoU is the average of the IoU of all semantic classes",id:"Mean Intersection over Union"},{description:"APα is the Average Precision at the IoU threshold of a α value, for example, AP50 and AP75",id:"APα"}],models:[{description:"Solid semantic segmentation model trained on ADE20k.",id:"openmmlab/upernet-convnext-small"},{description:"Background removal model.",id:"briaai/RMBG-1.4"},{description:"A multipurpose image segmentation model for high resolution images.",id:"ZhengPeng7/BiRefNet"},{description:"Powerful human-centric image segmentation model.",id:"facebook/sapiens-seg-1b"},{description:"Panoptic segmentation model trained on the COCO (common objects) dataset.",id:"facebook/mask2former-swin-large-coco-panoptic"}],spaces:[{description:"A semantic segmentation application that can predict unseen instances out of the box.",id:"facebook/ov-seg"},{description:"One of the strongest segmentation applications.",id:"jbrinkma/segment-anything"},{description:"A human-centric segmentation model.",id:"facebook/sapiens-pose"},{description:"An instance segmentation application to predict neuronal cell types from microscopy images.",id:"rashmi/sartorius-cell-instance-segmentation"},{description:"An application that segments videos.",id:"ArtGAN/Segment-Anything-Video"},{description:"An panoptic segmentation application built for outdoor environments.",id:"segments/panoptic-segment-anything"}],summary:"Image Segmentation divides an image into segments where each pixel in the image is mapped to an object. This task has multiple variants such as instance segmentation, panoptic segmentation and semantic segmentation.",widgetModels:["nvidia/segformer-b0-finetuned-ade-512-512"],youtubeId:"dKE8SIt9C-w"},Ot={datasets:[{description:"Widely used benchmark dataset for multiple Vision tasks.",id:"merve/coco2017"},{description:"Medical Imaging dataset of the Human Brain for segmentation and mask generating tasks",id:"rocky93/BraTS_segmentation"}],demo:{inputs:[{filename:"mask-generation-input.png",type:"img"}],outputs:[{filename:"mask-generation-output.png",type:"img"}]},metrics:[{description:"IoU is used to measure the overlap between predicted mask and the ground truth mask.",id:"Intersection over Union (IoU)"}],models:[{description:"Small yet powerful mask generation model.",id:"Zigeng/SlimSAM-uniform-50"},{description:"Very strong mask generation model.",id:"facebook/sam2-hiera-large"}],spaces:[{description:"An application that combines a mask generation model with a zero-shot object detection model for text-guided image segmentation.",id:"merve/OWLSAM2"},{description:"An application that compares the performance of a large and a small mask generation model.",id:"merve/slimsam"},{description:"An application based on an improved mask generation model.",id:"SkalskiP/segment-anything-model-2"},{description:"An application to remove objects from videos using mask generation models.",id:"SkalskiP/SAM_and_ProPainter"}],summary:"Mask generation is the task of generating masks that identify a specific object or region of interest in a given image. Masks are often used in segmentation tasks, where they provide a precise way to isolate the object of interest for further processing or analysis.",widgetModels:[],youtubeId:""},Mt={datasets:[{description:"Widely used benchmark dataset for multiple vision tasks.",id:"merve/coco2017"},{description:"Multi-task computer vision benchmark.",id:"merve/pascal-voc"}],demo:{inputs:[{filename:"object-detection-input.jpg",type:"img"}],outputs:[{filename:"object-detection-output.jpg",type:"img"}]},metrics:[{description:"The Average Precision (AP) metric is the Area Under the PR Curve (AUC-PR). It is calculated for each class separately",id:"Average Precision"},{description:"The Mean Average Precision (mAP) metric is the overall average of the AP values",id:"Mean Average Precision"},{description:"The APα metric is the Average Precision at the IoU threshold of a α value, for example, AP50 and AP75",id:"APα"}],models:[{description:"Solid object detection model pre-trained on the COCO 2017 dataset.",id:"facebook/detr-resnet-50"},{description:"Accurate object detection model.",id:"IDEA-Research/dab-detr-resnet-50"},{description:"Fast and accurate object detection model.",id:"PekingU/rtdetr_v2_r50vd"},{description:"Object detection model for low-lying objects.",id:"StephanST/WALDO30"}],spaces:[{description:"Leaderboard to compare various object detection models across several metrics.",id:"hf-vision/object_detection_leaderboard"},{description:"An application that contains various object detection models to try from.",id:"Gradio-Blocks/Object-Detection-With-DETR-and-YOLOS"},{description:"A cutting-edge object detection application.",id:"sunsmarterjieleaf/yolov12"},{description:"An object tracking, segmentation and inpainting application.",id:"VIPLab/Track-Anything"},{description:"Very fast object tracking application based on object detection.",id:"merve/RT-DETR-tracking-coco"}],summary:"Object Detection models allow users to identify objects of certain defined classes. Object detection models receive an image as input and output the images with bounding boxes and labels on detected objects.",widgetModels:["facebook/detr-resnet-50"],youtubeId:"WdAeKSOpxhw"},Lt={datasets:[{description:"NYU Depth V2 Dataset: Video dataset containing both RGB and depth sensor data.",id:"sayakpaul/nyu_depth_v2"},{description:"Monocular depth estimation benchmark based without noise and errors.",id:"depth-anything/DA-2K"}],demo:{inputs:[{filename:"depth-estimation-input.jpg",type:"img"}],outputs:[{filename:"depth-estimation-output.png",type:"img"}]},metrics:[],models:[{description:"Cutting-edge depth estimation model.",id:"depth-anything/Depth-Anything-V2-Large"},{description:"A strong monocular depth estimation model.",id:"jingheya/lotus-depth-g-v1-0"},{description:"A depth estimation model that predicts depth in videos.",id:"tencent/DepthCrafter"},{description:"A robust depth estimation model.",id:"apple/DepthPro-hf"}],spaces:[{description:"An application that predicts the depth of an image and then reconstruct the 3D model as voxels.",id:"radames/dpt-depth-estimation-3d-voxels"},{description:"An application for bleeding-edge depth estimation.",id:"akhaliq/depth-pro"},{description:"An application on cutting-edge depth estimation in videos.",id:"tencent/DepthCrafter"},{description:"A human-centric depth estimation application.",id:"facebook/sapiens-depth"}],summary:"Depth estimation is the task of predicting depth of the objects present in an image.",widgetModels:[""],youtubeId:""},re={datasets:[],demo:{inputs:[],outputs:[]},isPlaceholder:!0,metrics:[],models:[],spaces:[],summary:"",widgetModels:[],youtubeId:void 0,canonicalId:void 0},Dt={datasets:[{description:"A curation of widely used datasets for Data Driven Deep Reinforcement Learning (D4RL)",id:"edbeeching/decision_transformer_gym_replay"}],demo:{inputs:[{label:"State",content:"Red traffic light, pedestrians are about to pass.",type:"text"}],outputs:[{label:"Action",content:"Stop the car.",type:"text"},{label:"Next State",content:"Yellow light, pedestrians have crossed.",type:"text"}]},metrics:[{description:"Accumulated reward across all time steps discounted by a factor that ranges between 0 and 1 and determines how much the agent optimizes for future relative to immediate rewards. Measures how good is the policy ultimately found by a given algorithm considering uncertainty over the future.",id:"Discounted Total Reward"},{description:"Average return obtained after running the policy for a certain number of evaluation episodes. As opposed to total reward, mean reward considers how much reward a given algorithm receives while learning.",id:"Mean Reward"},{description:"Measures how good a given algorithm is after a predefined time. Some algorithms may be guaranteed to converge to optimal behavior across many time steps. However, an agent that reaches an acceptable level of optimality after a given time horizon may be preferable to one that ultimately reaches optimality but takes a long time.",id:"Level of Performance After Some Time"}],models:[{description:"A Reinforcement Learning model trained on expert data from the Gym Hopper environment",id:"edbeeching/decision-transformer-gym-hopper-expert"},{description:"A PPO agent playing seals/CartPole-v0 using the stable-baselines3 library and the RL Zoo.",id:"HumanCompatibleAI/ppo-seals-CartPole-v0"}],spaces:[{description:"An application for a cute puppy agent learning to catch a stick.",id:"ThomasSimonini/Huggy"},{description:"An application to play Snowball Fight with a reinforcement learning agent.",id:"ThomasSimonini/SnowballFight"}],summary:"Reinforcement learning is the computational approach of learning from action by interacting with an environment through trial and error and receiving rewards (negative or positive) as feedback",widgetModels:[],youtubeId:"q0BiUn5LiBc"},Nt={datasets:[{description:"A famous question answering dataset based on English articles from Wikipedia.",id:"squad_v2"},{description:"A dataset of aggregated anonymized actual queries issued to the Google search engine.",id:"natural_questions"}],demo:{inputs:[{label:"Question",content:"Which name is also used to describe the Amazon rainforest in English?",type:"text"},{label:"Context",content:"The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle",type:"text"}],outputs:[{label:"Answer",content:"Amazonia",type:"text"}]},metrics:[{description:"Exact Match is a metric based on the strict character match of the predicted answer and the right answer. For answers predicted correctly, the Exact Match will be 1. Even if only one character is different, Exact Match will be 0",id:"exact-match"},{description:" The F1-Score metric is useful if we value both false positives and false negatives equally. The F1-Score is calculated on each word in the predicted sequence against the correct answer",id:"f1"}],models:[{description:"A robust baseline model for most question answering domains.",id:"deepset/roberta-base-squad2"},{description:"Small yet robust model that can answer questions.",id:"distilbert/distilbert-base-cased-distilled-squad"},{description:"A special model that can answer questions from tables.",id:"google/tapas-base-finetuned-wtq"}],spaces:[{description:"An application that can answer a long question from Wikipedia.",id:"deepset/wikipedia-assistant"}],summary:"Question Answering models can retrieve the answer to a question from a given text, which is useful for searching for an answer in a document. Some question answering models can generate answers without context!",widgetModels:["deepset/roberta-base-squad2"],youtubeId:"ajPx5LwJD-I"},$t={datasets:[{description:"Bing queries with relevant passages from various web sources.",id:"microsoft/ms_marco"}],demo:{inputs:[{label:"Source sentence",content:"Machine learning is so easy.",type:"text"},{label:"Sentences to compare to",content:"Deep learning is so straightforward.",type:"text"},{label:"",content:"This is so difficult, like rocket science.",type:"text"},{label:"",content:"I can't believe how much I struggled with this.",type:"text"}],outputs:[{type:"chart",data:[{label:"Deep learning is so straightforward.",score:.623},{label:"This is so difficult, like rocket science.",score:.413},{label:"I can't believe how much I struggled with this.",score:.256}]}]},metrics:[{description:"Reciprocal Rank is a measure used to rank the relevancy of documents given a set of documents. Reciprocal Rank is the reciprocal of the rank of the document retrieved, meaning, if the rank is 3, the Reciprocal Rank is 0.33. If the rank is 1, the Reciprocal Rank is 1",id:"Mean Reciprocal Rank"},{description:"The similarity of the embeddings is evaluated mainly on cosine similarity. It is calculated as the cosine of the angle between two vectors. It is particularly useful when your texts are not the same length",id:"Cosine Similarity"}],models:[{description:"This model works well for sentences and paragraphs and can be used for clustering/grouping and semantic searches.",id:"sentence-transformers/all-mpnet-base-v2"},{description:"A multilingual robust sentence similarity model.",id:"BAAI/bge-m3"},{description:"A robust sentence similarity model.",id:"HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5"}],spaces:[{description:"An application that leverages sentence similarity to answer questions from YouTube videos.",id:"Gradio-Blocks/Ask_Questions_To_YouTube_Videos"},{description:"An application that retrieves relevant PubMed abstracts for a given online article which can be used as further references.",id:"Gradio-Blocks/pubmed-abstract-retriever"},{description:"An application that leverages sentence similarity to summarize text.",id:"nickmuchi/article-text-summarizer"},{description:"A guide that explains how Sentence Transformers can be used for semantic search.",id:"sentence-transformers/Sentence_Transformers_for_semantic_search"}],summary:"Sentence Similarity is the task of determining how similar two texts are. Sentence similarity models convert input texts into vectors (embeddings) that capture semantic information and calculate how close (similar) they are between them. This task is particularly useful for information retrieval and clustering/grouping.",widgetModels:["BAAI/bge-small-en-v1.5"],youtubeId:"VCZq5AkbNEU"},Rt={canonicalId:"text2text-generation",datasets:[{description:"News articles in five different languages along with their summaries. Widely used for benchmarking multilingual summarization models.",id:"mlsum"},{description:"English conversations and their summaries. Useful for benchmarking conversational agents.",id:"samsum"}],demo:{inputs:[{label:"Input",content:"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",type:"text"}],outputs:[{label:"Output",content:"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. It was the first structure to reach a height of 300 metres.",type:"text"}]},metrics:[{description:"The generated sequence is compared against its summary, and the overlap of tokens are counted. ROUGE-N refers to overlap of N subsequent tokens, ROUGE-1 refers to overlap of single tokens and ROUGE-2 is the overlap of two subsequent tokens.",id:"rouge"}],models:[{description:"A strong summarization model trained on English news articles. Excels at generating factual summaries.",id:"facebook/bart-large-cnn"},{description:"A summarization model trained on medical articles.",id:"Falconsai/medical_summarization"}],spaces:[{description:"An application that can summarize long paragraphs.",id:"pszemraj/summarize-long-text"},{description:"A much needed summarization application for terms and conditions.",id:"ml6team/distilbart-tos-summarizer-tosdr"},{description:"An application that summarizes long documents.",id:"pszemraj/document-summarization"},{description:"An application that can detect errors in abstractive summarization.",id:"ml6team/post-processing-summarization"}],summary:"Summarization is the task of producing a shorter version of a document while preserving its important information. Some models can extract text from the original input, while other models can generate entirely new text.",widgetModels:["facebook/bart-large-cnn"],youtubeId:"yHnr5Dk2zCI"},Pt={datasets:[{description:"The WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables.",id:"wikitablequestions"},{description:"WikiSQL is a dataset of 80654 hand-annotated examples of questions and SQL queries distributed across 24241 tables from Wikipedia.",id:"wikisql"}],demo:{inputs:[{table:[["Rank","Name","No.of reigns","Combined days"],["1","lou Thesz","3","3749"],["2","Ric Flair","8","3103"],["3","Harley Race","7","1799"]],type:"tabular"},{label:"Question",content:"What is the number of reigns for Harley Race?",type:"text"}],outputs:[{label:"Result",content:"7",type:"text"}]},metrics:[{description:"Checks whether the predicted answer(s) is the same as the ground-truth answer(s).",id:"Denotation Accuracy"}],models:[{description:"A table question answering model that is capable of neural SQL execution, i.e., employ TAPEX to execute a SQL query on a given table.",id:"microsoft/tapex-base"},{description:"A robust table question answering model.",id:"google/tapas-base-finetuned-wtq"}],spaces:[{description:"An application that answers questions based on table CSV files.",id:"katanaml/table-query"}],summary:"Table Question Answering (Table QA) is the answering a question about an information on a given table.",widgetModels:["google/tapas-base-finetuned-wtq"]},jt={datasets:[{description:"A comprehensive curation of datasets covering all benchmarks.",id:"inria-soda/tabular-benchmark"}],demo:{inputs:[{table:[["Glucose","Blood Pressure ","Skin Thickness","Insulin","BMI"],["148","72","35","0","33.6"],["150","50","30","0","35.1"],["141","60","29","1","39.2"]],type:"tabular"}],outputs:[{table:[["Diabetes"],["1"],["1"],["0"]],type:"tabular"}]},metrics:[{description:"",id:"accuracy"},{description:"",id:"recall"},{description:"",id:"precision"},{description:"",id:"f1"}],models:[{description:"Breast cancer prediction model based on decision trees.",id:"scikit-learn/cancer-prediction-trees"}],spaces:[{description:"An application that can predict defective products on a production line.",id:"scikit-learn/tabular-playground"},{description:"An application that compares various tabular classification techniques on different datasets.",id:"scikit-learn/classification"}],summary:"Tabular classification is the task of classifying a target category (a group) based on set of attributes.",widgetModels:["scikit-learn/tabular-playground"],youtubeId:""},Bt={datasets:[{description:"A comprehensive curation of datasets covering all benchmarks.",id:"inria-soda/tabular-benchmark"}],demo:{inputs:[{table:[["Car Name","Horsepower","Weight"],["ford torino","140","3,449"],["amc hornet","97","2,774"],["toyota corolla","65","1,773"]],type:"tabular"}],outputs:[{table:[["MPG (miles per gallon)"],["17"],["18"],["31"]],type:"tabular"}]},metrics:[{description:"",id:"mse"},{description:"Coefficient of determination (or R-squared) is a measure of how well the model fits the data. Higher R-squared is considered a better fit.",id:"r-squared"}],models:[{description:"Fish weight prediction based on length measurements and species.",id:"scikit-learn/Fish-Weight"}],spaces:[{description:"An application that can predict weight of a fish based on set of attributes.",id:"scikit-learn/fish-weight-prediction"}],summary:"Tabular regression is the task of predicting a numerical value given a set of attributes.",widgetModels:["scikit-learn/Fish-Weight"],youtubeId:""},qt={datasets:[{description:"RedCaps is a large-scale dataset of 12M image-text pairs collected from Reddit.",id:"red_caps"},{description:"Conceptual Captions is a dataset consisting of ~3.3M images annotated with captions.",id:"conceptual_captions"},{description:"12M image-caption pairs.",id:"Spawning/PD12M"}],demo:{inputs:[{label:"Input",content:"A city above clouds, pastel colors, Victorian style",type:"text"}],outputs:[{filename:"image.jpeg",type:"img"}]},metrics:[{description:"The Inception Score (IS) measure assesses diversity and meaningfulness. It uses a generated image sample to predict its label. A higher score signifies more diverse and meaningful images.",id:"IS"},{description:"The Fréchet Inception Distance (FID) calculates the distance between distributions between synthetic and real samples. A lower FID score indicates better similarity between the distributions of real and generated images.",id:"FID"},{description:"R-precision assesses how the generated image aligns with the provided text description. It uses the generated images as queries to retrieve relevant text descriptions. The top 'r' relevant descriptions are selected and used to calculate R-precision as r/R, where 'R' is the number of ground truth descriptions associated with the generated images. A higher R-precision value indicates a better model.",id:"R-Precision"}],models:[{description:"One of the most powerful image generation models that can generate realistic outputs.",id:"black-forest-labs/FLUX.1-dev"},{description:"A powerful yet fast image generation model.",id:"latent-consistency/lcm-lora-sdxl"},{description:"Text-to-image model for photorealistic generation.",id:"Kwai-Kolors/Kolors"},{description:"A powerful text-to-image model.",id:"stabilityai/stable-diffusion-3-medium-diffusers"}],spaces:[{description:"A powerful text-to-image application.",id:"stabilityai/stable-diffusion-3-medium"},{description:"A text-to-image application to generate comics.",id:"jbilcke-hf/ai-comic-factory"},{description:"An application to match multiple custom image generation models.",id:"multimodalart/flux-lora-lab"},{description:"A powerful yet very fast image generation application.",id:"latent-consistency/lcm-lora-for-sdxl"},{description:"A gallery to explore various text-to-image models.",id:"multimodalart/LoraTheExplorer"},{description:"An application for `text-to-image`, `image-to-image` and image inpainting.",id:"ArtGAN/Stable-Diffusion-ControlNet-WebUI"},{description:"An application to generate realistic images given photos of a person and a prompt.",id:"InstantX/InstantID"}],summary:"Text-to-image is the task of generating images from input text. These pipelines can also be used to modify and edit images based on text prompts.",widgetModels:["black-forest-labs/FLUX.1-dev"],youtubeId:""},Vt={canonicalId:"text-to-audio",datasets:[{description:"10K hours of multi-speaker English dataset.",id:"parler-tts/mls_eng_10k"},{description:"Multi-speaker English dataset.",id:"mythicinfinity/libritts_r"},{description:"Multi-lingual dataset.",id:"facebook/multilingual_librispeech"}],demo:{inputs:[{label:"Input",content:"I love audio models on the Hub!",type:"text"}],outputs:[{filename:"audio.wav",type:"audio"}]},metrics:[{description:"The Mel Cepstral Distortion (MCD) metric is used to calculate the quality of generated speech.",id:"mel cepstral distortion"}],models:[{description:"A prompt based, powerful TTS model.",id:"parler-tts/parler-tts-large-v1"},{description:"A powerful TTS model that supports English and Chinese.",id:"SWivid/F5-TTS"},{description:"A massively multi-lingual TTS model.",id:"fishaudio/fish-speech-1.5"},{description:"A powerful TTS model.",id:"OuteAI/OuteTTS-0.1-350M"},{description:"Small yet powerful TTS model.",id:"hexgrad/Kokoro-82M"}],spaces:[{description:"An application for generate high quality speech in different languages.",id:"hexgrad/Kokoro-TTS"},{description:"A multilingual text-to-speech application.",id:"fishaudio/fish-speech-1"},{description:"An application that generates speech in different styles in English and Chinese.",id:"mrfakename/E2-F5-TTS"},{description:"An application that synthesizes emotional speech for diverse speaker prompts.",id:"parler-tts/parler-tts-expresso"},{description:"An application that generates podcast episodes.",id:"ngxson/kokoro-podcast-generator"}],summary:"Text-to-Speech (TTS) is the task of generating natural sounding speech given text input. TTS models can be extended to have a single model that generates speech for multiple speakers and multiple languages.",widgetModels:["suno/bark"],youtubeId:"NW62DpzJ274"},Ft={datasets:[{description:"A widely used dataset useful to benchmark named entity recognition models.",id:"eriktks/conll2003"},{description:"A multilingual dataset of Wikipedia articles annotated for named entity recognition in over 150 different languages.",id:"unimelb-nlp/wikiann"}],demo:{inputs:[{label:"Input",content:"My name is Omar and I live in Zürich.",type:"text"}],outputs:[{text:"My name is Omar and I live in Zürich.",tokens:[{type:"PERSON",start:11,end:15},{type:"GPE",start:30,end:36}],type:"text-with-tokens"}]},metrics:[{description:"",id:"accuracy"},{description:"",id:"recall"},{description:"",id:"precision"},{description:"",id:"f1"}],models:[{description:"A robust performance model to identify people, locations, organizations and names of miscellaneous entities.",id:"dslim/bert-base-NER"},{description:"A strong model to identify people, locations, organizations and names in multiple languages.",id:"FacebookAI/xlm-roberta-large-finetuned-conll03-english"},{description:"A token classification model specialized on medical entity recognition.",id:"blaze999/Medical-NER"},{description:"Flair models are typically the state of the art in named entity recognition tasks.",id:"flair/ner-english"}],spaces:[{description:"An application that can recognizes entities, extracts noun chunks and recognizes various linguistic features of each token.",id:"spacy/gradio_pipeline_visualizer"}],summary:"Token classification is a natural language understanding task in which a label is assigned to some tokens in a text. Some popular token classification subtasks are Named Entity Recognition (NER) and Part-of-Speech (PoS) tagging. NER models could be trained to identify specific entities in a text, such as dates, individuals and places; and PoS tagging would identify, for example, which words in a text are verbs, nouns, and punctuation marks.",widgetModels:["FacebookAI/xlm-roberta-large-finetuned-conll03-english"],youtubeId:"wVHdVlPScxA"},zt={canonicalId:"text2text-generation",datasets:[{description:"A dataset of copyright-free books translated into 16 different languages.",id:"Helsinki-NLP/opus_books"},{description:"An example of translation between programming languages. This dataset consists of functions in Java and C#.",id:"google/code_x_glue_cc_code_to_code_trans"}],demo:{inputs:[{label:"Input",content:"My name is Omar and I live in Zürich.",type:"text"}],outputs:[{label:"Output",content:"Mein Name ist Omar und ich wohne in Zürich.",type:"text"}]},metrics:[{description:"BLEU score is calculated by counting the number of shared single or subsequent tokens between the generated sequence and the reference. Subsequent n tokens are called “n-grams”. Unigram refers to a single token while bi-gram refers to token pairs and n-grams refer to n subsequent tokens. The score ranges from 0 to 1, where 1 means the translation perfectly matched and 0 did not match at all",id:"bleu"},{description:"",id:"sacrebleu"}],models:[{description:"Very powerful model that can translate many languages between each other, especially low-resource languages.",id:"facebook/nllb-200-1.3B"},{description:"A general-purpose Transformer that can be used to translate from English to German, French, or Romanian.",id:"google-t5/t5-base"}],spaces:[{description:"An application that can translate between 100 languages.",id:"Iker/Translate-100-languages"},{description:"An application that can translate between many languages.",id:"Geonmo/nllb-translation-demo"}],summary:"Translation is the task of converting text from one language to another.",widgetModels:["facebook/mbart-large-50-many-to-many-mmt"],youtubeId:"1JvfrvZgi6c"},Ht={datasets:[{description:"A widely used dataset used to benchmark multiple variants of text classification.",id:"nyu-mll/glue"},{description:"A text classification dataset used to benchmark natural language inference models",id:"stanfordnlp/snli"}],demo:{inputs:[{label:"Input",content:"I love Hugging Face!",type:"text"}],outputs:[{type:"chart",data:[{label:"POSITIVE",score:.9},{label:"NEUTRAL",score:.1},{label:"NEGATIVE",score:0}]}]},metrics:[{description:"",id:"accuracy"},{description:"",id:"recall"},{description:"",id:"precision"},{description:"The F1 metric is the harmonic mean of the precision and recall. It can be calculated as: F1 = 2 * (precision * recall) / (precision + recall)",id:"f1"}],models:[{description:"A robust model trained for sentiment analysis.",id:"distilbert/distilbert-base-uncased-finetuned-sst-2-english"},{description:"A sentiment analysis model specialized in financial sentiment.",id:"ProsusAI/finbert"},{description:"A sentiment analysis model specialized in analyzing tweets.",id:"cardiffnlp/twitter-roberta-base-sentiment-latest"},{description:"A model that can classify languages.",id:"papluca/xlm-roberta-base-language-detection"},{description:"A model that can classify text generation attacks.",id:"meta-llama/Prompt-Guard-86M"}],spaces:[{description:"An application that can classify financial sentiment.",id:"IoannisTr/Tech_Stocks_Trading_Assistant"},{description:"A dashboard that contains various text classification tasks.",id:"miesnerjacob/Multi-task-NLP"},{description:"An application that analyzes user reviews in healthcare.",id:"spacy/healthsea-demo"}],summary:"Text Classification is the task of assigning a label or class to a given text. Some use cases are sentiment analysis, natural language inference, and assessing grammatical correctness.",widgetModels:["distilbert/distilbert-base-uncased-finetuned-sst-2-english"],youtubeId:"leNG9fN9FQU"},Wt={datasets:[{description:"Multilingual dataset used to evaluate text generation models.",id:"CohereForAI/Global-MMLU"},{description:"High quality multilingual data used to train text-generation models.",id:"HuggingFaceFW/fineweb-2"},{description:"Truly open-source, curated and cleaned dialogue dataset.",id:"HuggingFaceH4/ultrachat_200k"},{description:"A reasoning dataset.",id:"open-r1/OpenThoughts-114k-math"},{description:"A multilingual instruction dataset with preference ratings on responses.",id:"allenai/tulu-3-sft-mixture"},{description:"A large synthetic dataset for alignment of text generation models.",id:"HuggingFaceTB/smoltalk"},{description:"A dataset made for training text generation models solving math questions.",id:"HuggingFaceTB/finemath"}],demo:{inputs:[{label:"Input",content:"Once upon a time,",type:"text"}],outputs:[{label:"Output",content:"Once upon a time, we knew that our ancestors were on the verge of extinction. The great explorers and poets of the Old World, from Alexander the Great to Chaucer, are dead and gone. A good many of our ancient explorers and poets have",type:"text"}]},metrics:[{description:"Cross Entropy is a metric that calculates the difference between two probability distributions. Each probability distribution is the distribution of predicted words",id:"Cross Entropy"},{description:"The Perplexity metric is the exponential of the cross-entropy loss. It evaluates the probabilities assigned to the next word by the model. Lower perplexity indicates better performance",id:"Perplexity"}],models:[{description:"A text-generation model trained to follow instructions.",id:"google/gemma-2-2b-it"},{description:"Smaller variant of one of the most powerful models.",id:"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},{description:"Very powerful text generation model trained to follow instructions.",id:"meta-llama/Meta-Llama-3.1-8B-Instruct"},{description:"Powerful text generation model by Microsoft.",id:"microsoft/phi-4"},{description:"A very powerful model with reasoning capabilities.",id:"simplescaling/s1.1-32B"},{description:"Strong conversational model that supports very long instructions.",id:"Qwen/Qwen2.5-7B-Instruct-1M"},{description:"Text generation model used to write code.",id:"Qwen/Qwen2.5-Coder-32B-Instruct"},{description:"Powerful reasoning based open large language model.",id:"deepseek-ai/DeepSeek-R1"}],spaces:[{description:"A leaderboard to compare different open-source text generation models based on various benchmarks.",id:"open-llm-leaderboard/open_llm_leaderboard"},{description:"A leaderboard for comparing chain-of-thought performance of models.",id:"logikon/open_cot_leaderboard"},{description:"An text generation based application based on a very powerful LLaMA2 model.",id:"ysharma/Explore_llamav2_with_TGI"},{description:"An text generation based application to converse with Zephyr model.",id:"HuggingFaceH4/zephyr-chat"},{description:"A leaderboard that ranks text generation models based on blind votes from people.",id:"lmsys/chatbot-arena-leaderboard"},{description:"An chatbot to converse with a very powerful text generation model.",id:"mlabonne/phixtral-chat"}],summary:"Generating text is the task of generating new text given another text. These models can, for example, fill in incomplete text or paraphrase.",widgetModels:["mistralai/Mistral-Nemo-Instruct-2407"],youtubeId:"e9gNEAlsOvU"},Kt={datasets:[{description:"Bing queries with relevant passages from various web sources.",id:"microsoft/ms_marco"}],demo:{inputs:[{label:"Source sentence",content:"Machine learning is so easy.",type:"text"},{label:"Sentences to compare to",content:"Deep learning is so straightforward.",type:"text"},{label:"",content:"This is so difficult, like rocket science.",type:"text"},{label:"",content:"I can't believe how much I struggled with this.",type:"text"}],outputs:[{type:"chart",data:[{label:"Deep learning is so straightforward.",score:2.2006407},{label:"This is so difficult, like rocket science.",score:-6.2634873},{label:"I can't believe how much I struggled with this.",score:-10.251488}]}]},metrics:[{description:"Discounted Cumulative Gain (DCG) measures the gain, or usefulness, of search results discounted by their position. The normalization is done by dividing the DCG by the ideal DCG, which is the DCG of the perfect ranking.",id:"Normalized Discounted Cumulative Gain"},{description:"Reciprocal Rank is a measure used to rank the relevancy of documents given a set of documents. Reciprocal Rank is the reciprocal of the rank of the document retrieved, meaning, if the rank is 3, the Reciprocal Rank is 0.33. If the rank is 1, the Reciprocal Rank is 1",id:"Mean Reciprocal Rank"},{description:"Mean Average Precision (mAP) is the overall average of the Average Precision (AP) values, where AP is the Area Under the PR Curve (AUC-PR)",id:"Mean Average Precision"}],models:[{description:"An extremely efficient text ranking model trained on a web search dataset.",id:"cross-encoder/ms-marco-MiniLM-L6-v2"},{description:"A strong multilingual text reranker model.",id:"Alibaba-NLP/gte-multilingual-reranker-base"},{description:"An efficient text ranking model that punches above its weight.",id:"Alibaba-NLP/gte-reranker-modernbert-base"}],spaces:[],summary:"Text Ranking is the task of ranking a set of texts based on their relevance to a query. Text ranking models are trained on large datasets of queries and relevant documents to learn how to rank documents based on their relevance to the query. This task is particularly useful for search engines and information retrieval systems.",widgetModels:["cross-encoder/ms-marco-MiniLM-L6-v2"],youtubeId:""},Qt={datasets:[{description:"Microsoft Research Video to Text is a large-scale dataset for open domain video captioning",id:"iejMac/CLIP-MSR-VTT"},{description:"UCF101 Human Actions dataset consists of 13,320 video clips from YouTube, with 101 classes.",id:"quchenyuan/UCF101-ZIP"},{description:"A high-quality dataset for human action recognition in YouTube videos.",id:"nateraw/kinetics"},{description:"A dataset of video clips of humans performing pre-defined basic actions with everyday objects.",id:"HuggingFaceM4/something_something_v2"},{description:"This dataset consists of text-video pairs and contains noisy samples with irrelevant video descriptions",id:"HuggingFaceM4/webvid"},{description:"A dataset of short Flickr videos for the temporal localization of events with descriptions.",id:"iejMac/CLIP-DiDeMo"}],demo:{inputs:[{label:"Input",content:"Darth Vader is surfing on the waves.",type:"text"}],outputs:[{filename:"text-to-video-output.gif",type:"img"}]},metrics:[{description:"Inception Score uses an image classification model that predicts class labels and evaluates how distinct and diverse the images are. A higher score indicates better video generation.",id:"is"},{description:"Frechet Inception Distance uses an image classification model to obtain image embeddings. The metric compares mean and standard deviation of the embeddings of real and generated images. A smaller score indicates better video generation.",id:"fid"},{description:"Frechet Video Distance uses a model that captures coherence for changes in frames and the quality of each frame. A smaller score indicates better video generation.",id:"fvd"},{description:"CLIPSIM measures similarity between video frames and text using an image-text similarity model. A higher score indicates better video generation.",id:"clipsim"}],models:[{description:"A strong model for consistent video generation.",id:"tencent/HunyuanVideo"},{description:"A text-to-video model with high fidelity motion and strong prompt adherence.",id:"Lightricks/LTX-Video"},{description:"A text-to-video model focusing on physics-aware applications like robotics.",id:"nvidia/Cosmos-1.0-Diffusion-7B-Text2World"},{description:"A robust model for video generation.",id:"Wan-AI/Wan2.1-T2V-1.3B"}],spaces:[{description:"An application that generates video from text.",id:"VideoCrafter/VideoCrafter"},{description:"Consistent video generation application.",id:"Wan-AI/Wan2.1"},{description:"A cutting edge video generation application.",id:"Pyramid-Flow/pyramid-flow"}],summary:"Text-to-video models can be used in any application that requires generating consistent sequence of images from text. ",widgetModels:["Wan-AI/Wan2.1-T2V-14B"],youtubeId:void 0},Jt={datasets:[{description:"The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class.",id:"cifar100"},{description:"Multiple images of celebrities, used for facial expression translation.",id:"CelebA"}],demo:{inputs:[{label:"Seed",content:"42",type:"text"},{label:"Number of images to generate:",content:"4",type:"text"}],outputs:[{filename:"unconditional-image-generation-output.jpeg",type:"img"}]},metrics:[{description:"The inception score (IS) evaluates the quality of generated images. It measures the diversity of the generated images (the model predictions are evenly distributed across all possible labels) and their 'distinction' or 'sharpness' (the model confidently predicts a single label for each image).",id:"Inception score (IS)"},{description:"The Fréchet Inception Distance (FID) evaluates the quality of images created by a generative model by calculating the distance between feature vectors for real and generated images.",id:"Frećhet Inception Distance (FID)"}],models:[{description:"High-quality image generation model trained on the CIFAR-10 dataset. It synthesizes images of the ten classes presented in the dataset using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics.",id:"google/ddpm-cifar10-32"},{description:"High-quality image generation model trained on the 256x256 CelebA-HQ dataset. It synthesizes images of faces using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics.",id:"google/ddpm-celebahq-256"}],spaces:[{description:"An application that can generate realistic faces.",id:"CompVis/celeba-latent-diffusion"}],summary:"Unconditional image generation is the task of generating images with no condition in any context (like a prompt text or another image). Once trained, the model will create images that resemble its training data distribution.",widgetModels:[""],youtubeId:""},Xt={datasets:[{description:"Benchmark dataset used for video classification with videos that belong to 400 classes.",id:"kinetics400"}],demo:{inputs:[{filename:"video-classification-input.gif",type:"img"}],outputs:[{type:"chart",data:[{label:"Playing Guitar",score:.514},{label:"Playing Tennis",score:.193},{label:"Cooking",score:.068}]}]},metrics:[{description:"",id:"accuracy"},{description:"",id:"recall"},{description:"",id:"precision"},{description:"",id:"f1"}],models:[{description:"Strong Video Classification model trained on the Kinetics 400 dataset.",id:"google/vivit-b-16x2-kinetics400"},{description:"Strong Video Classification model trained on the Kinetics 400 dataset.",id:"microsoft/xclip-base-patch32"}],spaces:[{description:"An application that classifies video at different timestamps.",id:"nateraw/lavila"},{description:"An application that classifies video.",id:"fcakyon/video-classification"}],summary:"Video classification is the task of assigning a label or class to an entire video. Videos are expected to have only one class for each video. Video classification models take a video as input and return a prediction about which class the video belongs to.",widgetModels:[],youtubeId:""},Yt={datasets:[{description:"A widely used dataset containing questions (with answers) about images.",id:"Graphcore/vqa"},{description:"A dataset to benchmark visual reasoning based on text in images.",id:"facebook/textvqa"}],demo:{inputs:[{filename:"elephant.jpeg",type:"img"},{label:"Question",content:"What is in this image?",type:"text"}],outputs:[{type:"chart",data:[{label:"elephant",score:.97},{label:"elephants",score:.06},{label:"animal",score:.003}]}]},isPlaceholder:!1,metrics:[{description:"",id:"accuracy"},{description:"Measures how much a predicted answer differs from the ground truth based on the difference in their semantic meaning.",id:"wu-palmer similarity"}],models:[{description:"A visual question answering model trained to convert charts and plots to text.",id:"google/deplot"},{description:"A visual question answering model trained for mathematical reasoning and chart derendering from images.",id:"google/matcha-base"},{description:"A strong visual question answering that answers questions from book covers.",id:"google/pix2struct-ocrvqa-large"}],spaces:[{description:"An application that compares visual question answering models across different tasks.",id:"merve/pix2struct"},{description:"An application that can answer questions based on images.",id:"nielsr/vilt-vqa"},{description:"An application that can caption images and answer questions about a given image. ",id:"Salesforce/BLIP"},{description:"An application that can caption images and answer questions about a given image. ",id:"vumichien/Img2Prompt"}],summary:"Visual Question Answering is the task of answering open-ended questions based on an image. They output natural language responses to natural language questions.",widgetModels:["dandelin/vilt-b32-finetuned-vqa"],youtubeId:""},Gt={datasets:[{description:"A widely used dataset used to benchmark multiple variants of text classification.",id:"nyu-mll/glue"},{description:"The Multi-Genre Natural Language Inference (MultiNLI) corpus is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information.",id:"nyu-mll/multi_nli"},{description:"FEVER is a publicly available dataset for fact extraction and verification against textual sources.",id:"fever/fever"}],demo:{inputs:[{label:"Text Input",content:"Dune is the best movie ever.",type:"text"},{label:"Candidate Labels",content:"CINEMA, ART, MUSIC",type:"text"}],outputs:[{type:"chart",data:[{label:"CINEMA",score:.9},{label:"ART",score:.1},{label:"MUSIC",score:0}]}]},metrics:[],models:[{description:"Powerful zero-shot text classification model.",id:"facebook/bart-large-mnli"},{description:"Cutting-edge zero-shot multilingual text classification model.",id:"MoritzLaurer/ModernBERT-large-zeroshot-v2.0"},{description:"Zero-shot text classification model that can be used for topic and sentiment classification.",id:"knowledgator/gliclass-modern-base-v2.0-init"}],spaces:[],summary:"Zero-shot text classification is a task in natural language processing where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes.",widgetModels:["facebook/bart-large-mnli"]},Zt={datasets:[{description:"",id:""}],demo:{inputs:[{filename:"image-classification-input.jpeg",type:"img"},{label:"Classes",content:"cat, dog, bird",type:"text"}],outputs:[{type:"chart",data:[{label:"Cat",score:.664},{label:"Dog",score:.329},{label:"Bird",score:.008}]}]},metrics:[{description:"Computes the number of times the correct label appears in top K labels predicted",id:"top-K accuracy"}],models:[{description:"Multilingual image classification model for 80 languages.",id:"visheratin/mexma-siglip"},{description:"Strong zero-shot image classification model.",id:"google/siglip2-base-patch16-224"},{description:"Robust zero-shot image classification model.",id:"intfloat/mmE5-mllama-11b-instruct"},{description:"Powerful zero-shot image classification model supporting 94 languages.",id:"jinaai/jina-clip-v2"},{description:"Strong image classification model for biomedical domain.",id:"microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"}],spaces:[{description:"An application that leverages zero-shot image classification to find best captions to generate an image. ",id:"pharma/CLIP-Interrogator"},{description:"An application to compare different zero-shot image classification models. ",id:"merve/compare_clip_siglip"}],summary:"Zero-shot image classification is the task of classifying previously unseen classes during training of a model.",widgetModels:["google/siglip-so400m-patch14-224"],youtubeId:""},ei={datasets:[],demo:{inputs:[{filename:"zero-shot-object-detection-input.jpg",type:"img"},{label:"Classes",content:"cat, dog, bird",type:"text"}],outputs:[{filename:"zero-shot-object-detection-output.jpg",type:"img"}]},metrics:[{description:"The Average Precision (AP) metric is the Area Under the PR Curve (AUC-PR). It is calculated for each class separately",id:"Average Precision"},{description:"The Mean Average Precision (mAP) metric is the overall average of the AP values",id:"Mean Average Precision"},{description:"The APα metric is the Average Precision at the IoU threshold of a α value, for example, AP50 and AP75",id:"APα"}],models:[{description:"Solid zero-shot object detection model.",id:"IDEA-Research/grounding-dino-base"},{description:"Cutting-edge zero-shot object detection model.",id:"google/owlv2-base-patch16-ensemble"}],spaces:[{description:"A demo to try the state-of-the-art zero-shot object detection model, OWLv2.",id:"merve/owlv2"},{description:"A demo that combines a zero-shot object detection and mask generation model for zero-shot segmentation.",id:"merve/OWLSAM"}],summary:"Zero-shot object detection is a computer vision task to detect objects and their classes in images, without any prior training or knowledge of the classes. Zero-shot object detection models receive an image as input, as well as a list of candidate classes, and output the bounding boxes and labels where the objects have been detected.",widgetModels:[],youtubeId:""},ti={datasets:[{description:"A large dataset of over 10 million 3D objects.",id:"allenai/objaverse-xl"},{description:"A dataset of isolated object images for evaluating image-to-3D models.",id:"dylanebert/iso3d"}],demo:{inputs:[{filename:"image-to-3d-image-input.png",type:"img"}],outputs:[{label:"Result",content:"image-to-3d-3d-output-filename.glb",type:"text"}]},metrics:[],models:[{description:"Fast image-to-3D mesh model by Tencent.",id:"TencentARC/InstantMesh"},{description:"Fast image-to-3D mesh model by StabilityAI",id:"stabilityai/TripoSR"},{description:"A scaled up image-to-3D mesh model derived from TripoSR.",id:"hwjiang/Real3D"},{description:"Consistent image-to-3d generation model.",id:"stabilityai/stable-point-aware-3d"}],spaces:[{description:"Leaderboard to evaluate image-to-3D models.",id:"dylanebert/3d-arena"},{description:"Image-to-3D demo with mesh outputs.",id:"TencentARC/InstantMesh"},{description:"Image-to-3D demo.",id:"stabilityai/stable-point-aware-3d"},{description:"Image-to-3D demo with mesh outputs.",id:"hwjiang/Real3D"},{description:"Image-to-3D demo with splat outputs.",id:"dylanebert/LGM-mini"}],summary:"Image-to-3D models take in image input and produce 3D output.",widgetModels:[],youtubeId:""},ii={datasets:[{description:"A large dataset of over 10 million 3D objects.",id:"allenai/objaverse-xl"},{description:"Descriptive captions for 3D objects in Objaverse.",id:"tiange/Cap3D"}],demo:{inputs:[{label:"Prompt",content:"a cat statue",type:"text"}],outputs:[{label:"Result",content:"text-to-3d-3d-output-filename.glb",type:"text"}]},metrics:[],models:[{description:"Text-to-3D mesh model by OpenAI",id:"openai/shap-e"},{description:"Generative 3D gaussian splatting model.",id:"ashawkey/LGM"}],spaces:[{description:"Text-to-3D demo with mesh outputs.",id:"hysts/Shap-E"},{description:"Text/image-to-3D demo with splat outputs.",id:"ashawkey/LGM"}],summary:"Text-to-3D models take in text input and produce 3D output.",widgetModels:[],youtubeId:""},ai={datasets:[{description:"A dataset of hand keypoints of over 500k examples.",id:"Vincent-luo/hagrid-mediapipe-hands"}],demo:{inputs:[{filename:"keypoint-detection-input.png",type:"img"}],outputs:[{filename:"keypoint-detection-output.png",type:"img"}]},metrics:[],models:[{description:"A robust keypoint detection model.",id:"magic-leap-community/superpoint"},{description:"A robust keypoint matching model.",id:"magic-leap-community/superglue_outdoor"},{description:"Strong keypoint detection model used to detect human pose.",id:"facebook/sapiens-pose-1b"},{description:"Powerful keypoint detection model used to detect human pose.",id:"usyd-community/vitpose-plus-base"}],spaces:[{description:"An application that detects hand keypoints in real-time.",id:"datasciencedojo/Hand-Keypoint-Detection-Realtime"},{description:"An application to try a universal keypoint detection model.",id:"merve/SuperPoint"}],summary:"Keypoint detection is the task of identifying meaningful distinctive points or features in an image.",widgetModels:[],youtubeId:""},ni={datasets:[{description:"Multiple-choice questions and answers about videos.",id:"lmms-lab/Video-MME"},{description:"A dataset of instructions and question-answer pairs about videos.",id:"lmms-lab/VideoChatGPT"},{description:"Large video understanding dataset.",id:"HuggingFaceFV/finevideo"}],demo:{inputs:[{filename:"video-text-to-text-input.gif",type:"img"},{label:"Text Prompt",content:"What is happening in this video?",type:"text"}],outputs:[{label:"Answer",content:"The video shows a series of images showing a fountain with water jets and a variety of colorful flowers and butterflies in the background.",type:"text"}]},metrics:[],models:[{description:"A robust video-text-to-text model.",id:"Vision-CAIR/LongVU_Qwen2_7B"},{description:"Strong video-text-to-text model with reasoning capabilities.",id:"GoodiesHere/Apollo-LMMs-Apollo-7B-t32"},{description:"Strong video-text-to-text model.",id:"HuggingFaceTB/SmolVLM2-2.2B-Instruct"}],spaces:[{description:"An application to chat with a video-text-to-text model.",id:"llava-hf/video-llava"},{description:"A leaderboard for various video-text-to-text models.",id:"opencompass/openvlm_video_leaderboard"},{description:"An application to generate highlights from a video.",id:"HuggingFaceTB/SmolVLM2-HighlightGenerator"}],summary:"Video-text-to-text models take in a video and a text prompt and output text. These models are also called video-language models.",widgetModels:[""],youtubeId:""},oi={"audio-classification":["speechbrain","transformers","transformers.js"],"audio-to-audio":["asteroid","fairseq","speechbrain"],"automatic-speech-recognition":["espnet","nemo","speechbrain","transformers","transformers.js"],"audio-text-to-text":[],"depth-estimation":["transformers","transformers.js"],"document-question-answering":["transformers","transformers.js"],"feature-extraction":["sentence-transformers","transformers","transformers.js"],"fill-mask":["transformers","transformers.js"],"graph-ml":["transformers"],"image-classification":["keras","timm","transformers","transformers.js"],"image-feature-extraction":["timm","transformers"],"image-segmentation":["transformers","transformers.js"],"image-text-to-text":["transformers"],"image-to-image":["diffusers","transformers","transformers.js"],"image-to-text":["transformers","transformers.js"],"image-to-video":["diffusers"],"keypoint-detection":["transformers"],"video-classification":["transformers"],"mask-generation":["transformers"],"multiple-choice":["transformers"],"object-detection":["transformers","transformers.js","ultralytics"],other:[],"question-answering":["adapter-transformers","allennlp","transformers","transformers.js"],robotics:[],"reinforcement-learning":["transformers","stable-baselines3","ml-agents","sample-factory"],"sentence-similarity":["sentence-transformers","spacy","transformers.js"],summarization:["transformers","transformers.js"],"table-question-answering":["transformers"],"table-to-text":["transformers"],"tabular-classification":["sklearn"],"tabular-regression":["sklearn"],"tabular-to-text":["transformers"],"text-classification":["adapter-transformers","setfit","spacy","transformers","transformers.js"],"text-generation":["transformers","transformers.js"],"text-ranking":["sentence-transformers","transformers"],"text-retrieval":[],"text-to-image":["diffusers"],"text-to-speech":["espnet","tensorflowtts","transformers","transformers.js"],"text-to-audio":["transformers","transformers.js"],"text-to-video":["diffusers"],"text2text-generation":["transformers","transformers.js"],"time-series-forecasting":[],"token-classification":["adapter-transformers","flair","spacy","span-marker","stanza","transformers","transformers.js"],translation:["transformers","transformers.js"],"unconditional-image-generation":["diffusers"],"video-text-to-text":["transformers"],"visual-question-answering":["transformers","transformers.js"],"voice-activity-detection":[],"zero-shot-classification":["transformers","transformers.js"],"zero-shot-image-classification":["transformers","transformers.js"],"zero-shot-object-detection":["transformers","transformers.js"],"text-to-3d":["diffusers"],"image-to-3d":["diffusers"],"any-to-any":["transformers"],"visual-document-retrieval":["transformers"]};function w(e,i=re){return{...i,id:e,label:N[e].name,libraries:oi[e]}}w("any-to-any",re),w("audio-classification",wt),w("audio-to-audio",vt),w("audio-text-to-text",re),w("automatic-speech-recognition",_t),w("depth-estimation",Lt),w("document-question-answering",kt),w("visual-document-retrieval",re),w("feature-extraction",xt),w("fill-mask",At),w("image-classification",St),w("image-feature-extraction",It),w("image-segmentation",Ut),w("image-to-image",Et),w("image-text-to-text",Ct),w("image-to-text",Tt),w("keypoint-detection",ai),w("mask-generation",Ot),w("object-detection",Mt),w("video-classification",Xt),w("question-answering",Nt),w("reinforcement-learning",Dt),w("sentence-similarity",$t),w("summarization",Rt),w("table-question-answering",Pt),w("tabular-classification",jt),w("tabular-regression",Bt),w("text-classification",Ht),w("text-generation",Wt),w("text-ranking",Kt),w("text-to-image",qt),w("text-to-speech",Vt),w("text-to-video",Qt),w("token-classification",Ft),w("translation",zt),w("unconditional-image-generation",Jt),w("video-text-to-text",ni),w("visual-question-answering",Yt),w("zero-shot-classification",Gt),w("zero-shot-image-classification",Zt),w("zero-shot-object-detection",ei),w("text-to-3d",ii),w("image-to-3d",ti);const ri=()=>'"Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"',si=()=>'"Меня зовут Вольфганг и я живу в Берлине"',li=()=>'"The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."',ci=()=>`{
    "query": "How many stars does the transformers repository have?",
    "table": {
        "Repository": ["Transformers", "Datasets", "Tokenizers"],
        "Stars": ["36542", "4512", "3934"],
        "Contributors": ["651", "77", "34"],
        "Programming language": [
            "Python",
            "Python",
            "Rust, Python and NodeJS"
        ]
    }
}`,pi=()=>`{
        "image": "cat.png",
        "question": "What is in this image?"
    }`,di=()=>`{
    "question": "What is my name?",
    "context": "My name is Clara and I live in Berkeley."
}`,ui=()=>'"I like you. I love you"',mi=()=>'"My name is Sarah Jessica Parker but you can call me Jessica"',Ee=e=>e.tags.includes("conversational")?e.pipeline_tag==="text-generation"?[{role:"user",content:"What is the capital of France?"}]:[{role:"user",content:[{type:"text",text:"Describe this image in one sentence."},{type:"image_url",image_url:{url:"https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}]}]:'"Can you please let us know more details about your "',fi=()=>'"The answer to the universe is"',hi=e=>`"The answer to the universe is ${e.mask_token}."`,gi=()=>`{
    "source_sentence": "That is a happy person",
    "sentences": [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
}`,bi=()=>'"Today is a sunny day and I will get some ice cream."',yi=()=>'"cats.jpg"',wi=()=>'"cats.jpg"',vi=()=>`{
    "image": "cat.png",
    "prompt": "Turn the cat into a tiger."
}`,_i=()=>'"cats.jpg"',ki=()=>'"cats.jpg"',xi=()=>'"sample1.flac"',Ai=()=>'"sample1.flac"',Si=()=>'"Astronaut riding a horse"',Ii=()=>'"A young man walking on the street"',Ei=()=>'"The answer to the universe is 42"',Ti=()=>'"liquid drum and bass, atmospheric synths, airy sounds"',Ci=()=>'"sample1.flac"',Te=()=>`'{"Height":[11.52,12.48],"Length1":[23.2,24.0],"Length2":[25.4,26.3],"Species": ["Bream","Bream"]}'`,Ui={"audio-to-audio":xi,"audio-classification":Ai,"automatic-speech-recognition":Ci,"document-question-answering":pi,"feature-extraction":bi,"fill-mask":hi,"image-classification":yi,"image-to-text":wi,"image-to-image":vi,"image-segmentation":_i,"object-detection":ki,"question-answering":di,"sentence-similarity":gi,summarization:li,"table-question-answering":ci,"tabular-regression":Te,"tabular-classification":Te,"text-classification":ui,"text-generation":Ee,"image-text-to-text":Ee,"text-to-image":Si,"text-to-video":Ii,"text-to-speech":Ei,"text-to-audio":Ti,"text2text-generation":fi,"token-classification":mi,translation:si,"zero-shot-classification":ri,"zero-shot-image-classification":()=>'"cats.jpg"'};function te(e,i=!1,t=!1){if(e.pipeline_tag){const a=Ui[e.pipeline_tag];if(a){let n=a(e);if(typeof n=="string"&&(i&&(n=n.replace(/(?:(?:\r?\n|\r)\t*)|\t+/g," ")),t)){const o=/^"(.+)"$/s,s=n.match(o);n=s?s[1]:n}return n}}return"No input example has been defined for this model task."}function Oi(e,i){let t=JSON.stringify(e,null,"	");return i!=null&&i.indent&&(t=t.replaceAll(`
`,`
${i.indent}`)),i!=null&&i.attributeKeyQuotes||(t=t.replace(/"([^"]+)":/g,"$1:")),i!=null&&i.customContentEscaper&&(t=i.customContentEscaper(t)),t}const Ce="custom_code";function V(e){const i=e.split("/");return i.length===1?i[0]:i[1]}const Mi=e=>JSON.stringify(e).slice(1,-1),Li=e=>{var i,t;return[`from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("${(t=(i=e.config)==null?void 0:i.adapter_transformers)==null?void 0:t.model_name}")
model.load_adapter("${e.id}", set_active=True)`]},Di=e=>[`import allennlp_models
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("hf://${e.id}")`],Ni=e=>[`import allennlp_models
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("hf://${e.id}")
predictor_input = {"passage": "My name is Wolfgang and I live in Berlin", "question": "Where do I live?"}
predictions = predictor.predict_json(predictor_input)`],$i=e=>e.tags.includes("question-answering")?Ni(e):Di(e),Ri=e=>[`from araclip import AraClip

model = AraClip.from_pretrained("${e.id}")`],Pi=e=>[`from asteroid.models import BaseModel

model = BaseModel.from_pretrained("${e.id}")`],ji=e=>{const i=`# Watermark Generator
from audioseal import AudioSeal

model = AudioSeal.load_generator("${e.id}")
# pass a tensor (tensor_wav) of shape (batch, channels, samples) and a sample rate
wav, sr = tensor_wav, 16000
	
watermark = model.get_watermark(wav, sr)
watermarked_audio = wav + watermark`,t=`# Watermark Detector
from audioseal import AudioSeal

detector = AudioSeal.load_detector("${e.id}")
	
result, message = detector.detect_watermark(watermarked_audio, sr)`;return[i,t]};function fe(e){var i,t;return((t=(i=e.cardData)==null?void 0:i.base_model)==null?void 0:t.toString())??"fill-in-base-model"}function Ue(e){var t,a,n;const i=((a=(t=e.widgetData)==null?void 0:t[0])==null?void 0:a.text)??((n=e.cardData)==null?void 0:n.instance_prompt);if(i)return Mi(i)}const Bi=e=>[`import requests
from PIL import Image
from ben2 import AutoModel

url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = AutoModel.from_pretrained("${e.id}")
model.to("cuda").eval()
foreground = model.inference(image)
`],qi=e=>[`from bertopic import BERTopic

model = BERTopic.load("${e.id}")`],Vi=e=>[`from bm25s.hf import BM25HF

retriever = BM25HF.load_from_hub("${e.id}")`],Fi=()=>[`# pip install git+https://github.com/Google-Health/cxr-foundation.git#subdirectory=python

# Load image as grayscale (Stillwaterising, CC0, via Wikimedia Commons)
import requests
from PIL import Image
from io import BytesIO
image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
img = Image.open(requests.get(image_url, headers={'User-Agent': 'Demo'}, stream=True).raw).convert('L')

# Run inference
from clientside.clients import make_hugging_face_client
cxr_client = make_hugging_face_client('cxr_model')
print(cxr_client.get_image_embeddings_from_images([img]))`],zi=e=>{let i,t,a;return i="<ENCODER>",t="<NUMBER_OF_FEATURES>",a="<OUT_CHANNELS>",e.id==="depth-anything/Depth-Anything-V2-Small"?(i="vits",t="64",a="[48, 96, 192, 384]"):e.id==="depth-anything/Depth-Anything-V2-Base"?(i="vitb",t="128",a="[96, 192, 384, 768]"):e.id==="depth-anything/Depth-Anything-V2-Large"&&(i="vitl",t="256",a="[256, 512, 1024, 1024"),[`
# Install from https://github.com/DepthAnything/Depth-Anything-V2

# Load the model and infer depth from an image
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# instantiate the model
model = DepthAnythingV2(encoder="${i}", features=${t}, out_channels=${a})

# load the weights
filepath = hf_hub_download(repo_id="${e.id}", filename="depth_anything_v2_${i}.pth", repo_type="model")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict(state_dict).eval()

raw_img = cv2.imread("your/image/path")
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    `]},Hi=e=>[`# Download checkpoint
pip install huggingface-hub
huggingface-cli download --local-dir checkpoints ${e.id}`,`import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb("example.png")
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)

# Results: 1. Depth in meters
depth = prediction["depth"]
# Results: 2. Focal length in pixels
focallength_px = prediction["focallength_px"]`],Wi=()=>[`from huggingface_hub import from_pretrained_keras
import tensorflow as tf, requests

# Load and format input
IMAGE_URL = "https://storage.googleapis.com/dx-scin-public-data/dataset/images/3445096909671059178.png"
input_tensor = tf.train.Example(
    features=tf.train.Features(
        feature={
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[requests.get(IMAGE_URL, stream=True).content])
            )
        }
    )
).SerializeToString()

# Load model and run inference
loaded_model = from_pretrained_keras("google/derm-foundation")
infer = loaded_model.signatures["serving_default"]
print(infer(inputs=tf.constant([input_tensor])))`],Oe="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",Ki=e=>[`from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("${e.id}")

prompt = "${Ue(e)??Oe}"
image = pipe(prompt).images[0]`],Qi=e=>[`from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained("${e.id}")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"${fe(e)}", controlnet=controlnet
)`],Ji=e=>[`from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("${fe(e)}")
pipe.load_lora_weights("${e.id}")

prompt = "${Ue(e)??Oe}"
image = pipe(prompt).images[0]`],Xi=e=>[`from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("${fe(e)}")
pipe.load_textual_inversion("${e.id}")`],Yi=e=>e.tags.includes("controlnet")?Qi(e):e.tags.includes("lora")?Ji(e):e.tags.includes("textual_inversion")?Xi(e):Ki(e),Gi=e=>{const i=`# Pipeline for Stable Diffusion 3
from diffusionkit.mlx import DiffusionPipeline

pipeline = DiffusionPipeline(
	shift=3.0,
	use_t5=False,
	model_version=${e.id},
	low_memory_mode=True,
	a16=True,
	w16=True,
)`,t=`# Pipeline for Flux
from diffusionkit.mlx import FluxPipeline

pipeline = FluxPipeline(
  shift=1.0,
  model_version=${e.id},
  low_memory_mode=True,
  a16=True,
  w16=True,
)`,a=`# Image Generation
HEIGHT = 512
WIDTH = 512
NUM_STEPS = ${e.tags.includes("flux")?4:50}
CFG_WEIGHT = ${e.tags.includes("flux")?0:5}

image, _ = pipeline.generate_image(
  "a photo of a cat",
  cfg_weight=CFG_WEIGHT,
  num_steps=NUM_STEPS,
  latent_size=(HEIGHT // 8, WIDTH // 8),
)`;return[e.tags.includes("flux")?t:i,a]},Zi=e=>[`# pip install --no-binary :all: cartesia-pytorch
from cartesia_pytorch import ReneLMHeadModel
from transformers import AutoTokenizer

model = ReneLMHeadModel.from_pretrained("${e.id}")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

in_message = ["Rene Descartes was"]
inputs = tokenizer(in_message, return_tensors="pt")

outputs = model.generate(inputs.input_ids, max_length=50, top_k=100, top_p=0.99)
out_message = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(out_message)
)`],ea=e=>[`import mlx.core as mx
import cartesia_mlx as cmx

model = cmx.from_pretrained("${e.id}")
model.set_dtype(mx.float32)   

prompt = "Rene Descartes was"

for text in model.generate(
    prompt,
    max_tokens=500,
    eval_every_n=5,
    verbose=True,
    top_p=0.99,
    temperature=0.85,
):
    print(text, end="", flush=True)
`],ta=e=>{const i=V(e.id).replaceAll("-","_");return[`# Load it from the Hub directly
import edsnlp
nlp = edsnlp.load("${e.id}")
`,`# Or install it as a package
!pip install git+https://huggingface.co/${e.id}

# and import it as a module
import ${i}

nlp = ${i}.load()  # or edsnlp.load("${i}")
`]},ia=e=>[`from espnet2.bin.tts_inference import Text2Speech

model = Text2Speech.from_pretrained("${e.id}")

speech, *_ = model("text to generate speech from")`],aa=e=>[`from espnet2.bin.asr_inference import Speech2Text

model = Speech2Text.from_pretrained(
  "${e.id}"
)

speech, rate = soundfile.read("speech.wav")
text, *_ = model(speech)[0]`],na=()=>["unknown model type (must be text-to-speech or automatic-speech-recognition)"],oa=e=>e.tags.includes("text-to-speech")?ia(e):e.tags.includes("automatic-speech-recognition")?aa(e):na(),ra=e=>[`from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "${e.id}"
)`],sa=e=>[`from flair.models import SequenceTagger

tagger = SequenceTagger.load("${e.id}")`],la=e=>[`from gliner import GLiNER

model = GLiNER.from_pretrained("${e.id}")`],ca=e=>[`# CLI usage
# see docs: https://ai-riksarkivet.github.io/htrflow/latest/getting_started/quick_start.html
htrflow pipeline <path/to/pipeline.yaml> <path/to/image>`,`# Python usage
from htrflow.pipeline.pipeline import Pipeline
from htrflow.pipeline.steps import Task
from htrflow.models.framework.model import ModelClass

pipeline = Pipeline(
    [
        Task(
            ModelClass, {"model": "${e.id}"}, {}
        ),
    ])`],pa=e=>[`# Available backend options are: "jax", "torch", "tensorflow".
import os
os.environ["KERAS_BACKEND"] = "jax"
	
import keras

model = keras.saving.load_model("hf://${e.id}")
`],Me={CausalLM:e=>`
import keras_hub

# Load CausalLM model (optional: use half precision for inference)
causal_lm = keras_hub.models.CausalLM.from_preset("hf://${e}", dtype="bfloat16")
causal_lm.compile(sampler="greedy")  # (optional) specify a sampler

# Generate text
causal_lm.generate("Keras: deep learning for", max_length=64)
`,TextToImage:e=>`
import keras_hub

# Load TextToImage model (optional: use half precision for inference)
text_to_image = keras_hub.models.TextToImage.from_preset("hf://${e}", dtype="bfloat16")

# Generate images with a TextToImage model.
text_to_image.generate("Astronaut in a jungle")
`,TextClassifier:e=>`
import keras_hub

# Load TextClassifier model
text_classifier = keras_hub.models.TextClassifier.from_preset(
    "hf://${e}",
    num_classes=2,
)
# Fine-tune
text_classifier.fit(x=["Thilling adventure!", "Total snoozefest."], y=[1, 0])
# Classify text
text_classifier.predict(["Not my cup of tea."])
`,ImageClassifier:e=>`
import keras_hub
import keras

# Load ImageClassifier model
image_classifier = keras_hub.models.ImageClassifier.from_preset(
    "hf://${e}",
    num_classes=2,
)
# Fine-tune
image_classifier.fit(
    x=keras.random.randint((32, 64, 64, 3), 0, 256),
    y=keras.random.randint((32, 1), 0, 2),
)
# Classify image
image_classifier.predict(keras.random.randint((1, 64, 64, 3), 0, 256))
`},da=(e,i)=>`
import keras_hub

# Create a ${e} model
task = keras_hub.models.${e}.from_preset("hf://${i}")
`,ua=e=>`
import keras_hub

# Create a Backbone model unspecialized for any task
backbone = keras_hub.models.Backbone.from_preset("hf://${e}")
`,ma=e=>{var n,o;const i=e.id,t=((o=(n=e.config)==null?void 0:n.keras_hub)==null?void 0:o.tasks)??[],a=[];for(const[s,d]of Object.entries(Me))t.includes(s)&&a.push(d(i));for(const s of t)Object.keys(Me).includes(s)||a.push(da(s,i));return a.push(ua(i)),a},fa=e=>{const i=[`from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="${e.id}",
	filename="{{GGUF_FILE}}",
)
`];if(e.tags.includes("conversational")){const t=te(e);i.push(`llm.create_chat_completion(
	messages = ${Oi(t,{attributeKeyQuotes:!0,indent:"	"})}
)`)}else i.push(`output = llm(
	"Once upon a time,",
	max_tokens=512,
	echo=True
)
print(output)`);return i},ha=e=>[`# Note: 'keras<3.x' or 'tf_keras' must be installed (legacy)
# See https://github.com/keras-team/tf-keras for more details.
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("${e.id}")
`],ga=e=>[`from mamba_ssm import MambaLMHeadModel

model = MambaLMHeadModel.from_pretrained("${e.id}")`],ba=e=>[`# Install from https://github.com/Camb-ai/MARS5-TTS

from inference import Mars5TTS
mars5 = Mars5TTS.from_pretrained("${e.id}")`],ya=e=>[`# Install from https://github.com/pq-yang/MatAnyone.git

from matanyone.model.matanyone import MatAnyone
model = MatAnyone.from_pretrained("${e.id}")`],wa=()=>[`# Install from https://github.com/buaacyw/MeshAnything.git

from MeshAnything.models.meshanything import MeshAnything

# refer to https://github.com/buaacyw/MeshAnything/blob/main/main.py#L91 on how to define args
# and https://github.com/buaacyw/MeshAnything/blob/main/app.py regarding usage
model = MeshAnything(args)`],va=e=>[`import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:${e.id}')
tokenizer = open_clip.get_tokenizer('hf-hub:${e.id}')`],_a=e=>{var i,t;if((t=(i=e.config)==null?void 0:i.architectures)!=null&&t[0]){const a=e.config.architectures[0];return[[`from paddlenlp.transformers import AutoTokenizer, ${a}`,"",`tokenizer = AutoTokenizer.from_pretrained("${e.id}", from_hf_hub=True)`,`model = ${a}.from_pretrained("${e.id}", from_hf_hub=True)`].join(`
`)]}else return[["# ⚠️ Type of model unknown","from paddlenlp.transformers import AutoTokenizer, AutoModel","",`tokenizer = AutoTokenizer.from_pretrained("${e.id}", from_hf_hub=True)`,`model = AutoModel.from_pretrained("${e.id}", from_hf_hub=True)`].join(`
`)]},ka=e=>[`from pyannote.audio import Pipeline
  
pipeline = Pipeline.from_pretrained("${e.id}")

# inference on the whole file
pipeline("file.wav")

# inference on an excerpt
from pyannote.core import Segment
excerpt = Segment(start=2.0, end=5.0)

from pyannote.audio import Audio
waveform, sample_rate = Audio().crop("file.wav", excerpt)
pipeline({"waveform": waveform, "sample_rate": sample_rate})`],xa=e=>[`from pyannote.audio import Model, Inference

model = Model.from_pretrained("${e.id}")
inference = Inference(model)

# inference on the whole file
inference("file.wav")

# inference on an excerpt
from pyannote.core import Segment
excerpt = Segment(start=2.0, end=5.0)
inference.crop("file.wav", excerpt)`],Aa=e=>e.tags.includes("pyannote-audio-pipeline")?ka(e):xa(e),Sa=e=>[`from relik import Relik
 
relik = Relik.from_pretrained("${e.id}")`],Ia=e=>[`from tensorflow_tts.inference import AutoProcessor, TFAutoModel

processor = AutoProcessor.from_pretrained("${e.id}")
model = TFAutoModel.from_pretrained("${e.id}")
`],Ea=e=>[`from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("${e.id}")
audios = model.inference(mels)
`],Ta=e=>[`from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("${e.id}")
`],Ca=e=>e.tags.includes("text-to-mel")?Ia(e):e.tags.includes("mel-to-wav")?Ea(e):Ta(e),Ua=e=>[`import timm

model = timm.create_model("hf_hub:${e.id}", pretrained=True)`],Oa=()=>[`# pip install sae-lens
from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "RELEASE_ID", # e.g., "gpt2-small-res-jb". See other options in https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
    sae_id = "SAE_ID", # e.g., "blocks.8.hook_resid_pre". Won't always be a hook point
)`],Ma=()=>[`# seed_story_cfg_path refers to 'https://github.com/TencentARC/SEED-Story/blob/master/configs/clm_models/agent_7b_sft.yaml'
# llm_cfg_path refers to 'https://github.com/TencentARC/SEED-Story/blob/master/configs/clm_models/llama2chat7b_lora.yaml'
from omegaconf import OmegaConf
import hydra

# load Llama2
llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype="fp16")

# initialize seed_story
seed_story_cfg = OmegaConf.load(seed_story_cfg_path)
seed_story = hydra.utils.instantiate(seed_story_cfg, llm=llm) `],La=(e,i)=>[`import joblib
from skops.hub_utils import download
download("${e.id}", "path_to_folder")
model = joblib.load(
	"${i}"
)
# only load pickle files from sources you trust
# read more about it here https://skops.readthedocs.io/en/stable/persistence.html`],Da=(e,i)=>[`from skops.hub_utils import download
from skops.io import load
download("${e.id}", "path_to_folder")
# make sure model file is in skops format
# if model is a pickle file, make sure it's from a source you trust
model = load("path_to_folder/${i}")`],Na=e=>[`from huggingface_hub import hf_hub_download
import joblib
model = joblib.load(
	hf_hub_download("${e.id}", "sklearn_model.joblib")
)
# only load pickle files from sources you trust
# read more about it here https://skops.readthedocs.io/en/stable/persistence.html`],$a=e=>{var i,t,a,n,o;if(e.tags.includes("skops")){const s=(a=(t=(i=e.config)==null?void 0:i.sklearn)==null?void 0:t.model)==null?void 0:a.file,d=(o=(n=e.config)==null?void 0:n.sklearn)==null?void 0:o.model_format;return s?d==="pickle"?La(e,s):Da(e,s):["# ⚠️ Model filename not specified in config.json"]}else return Na(e)},Ra=e=>[`import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("${e.id}")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
	"prompt": "128 BPM tech house drum loop",
}]

# Generate stereo audio
output = generate_diffusion_cond(
	model,
	conditioning=conditioning,
	sample_size=sample_size,
	device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)`],Pa=e=>[`from huggingface_hub import from_pretrained_fastai

learn = from_pretrained_fastai("${e.id}")`],ja=e=>{const i=`# Use SAM2 with images
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained(${e.id})

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)`,t=`# Use SAM2 with videos
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
	
predictor = SAM2VideoPredictor.from_pretrained(${e.id})

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...`;return[i,t]},Ba=e=>[`python -m sample_factory.huggingface.load_from_hub -r ${e.id} -d ./train_dir`];function qa(e){var t;const i=(t=e.widgetData)==null?void 0:t[0];if(i)return[i.source_sentence,...i.sentences]}const Va=e=>{const i=e.tags.includes(Ce)?", trust_remote_code=True":"",t=qa(e)??["The weather is lovely today.","It's so sunny outside!","He drove to the stadium."];return[`from sentence_transformers import SentenceTransformer

model = SentenceTransformer("${e.id}"${i})

sentences = ${JSON.stringify(t,null,4)}
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [${t.length}, ${t.length}]`]},Fa=e=>[`from setfit import SetFitModel

model = SetFitModel.from_pretrained("${e.id}")`],za=e=>[`!pip install https://huggingface.co/${e.id}/resolve/main/${V(e.id)}-any-py3-none-any.whl

# Using spacy.load().
import spacy
nlp = spacy.load("${V(e.id)}")

# Importing as module.
import ${V(e.id)}
nlp = ${V(e.id)}.load()`],Ha=e=>[`from span_marker import SpanMarkerModel

model = SpanMarkerModel.from_pretrained("${e.id}")`],Wa=e=>[`import stanza

stanza.download("${V(e.id).replace("stanza-","")}")
nlp = stanza.Pipeline("${V(e.id).replace("stanza-","")}")`],Ka=e=>{switch(e){case"EncoderClassifier":return"classify_file";case"EncoderDecoderASR":case"EncoderASR":return"transcribe_file";case"SpectralMaskEnhancement":return"enhance_file";case"SepformerSeparation":return"separate_file";default:return}},Qa=e=>{var a,n;const i=(n=(a=e.config)==null?void 0:a.speechbrain)==null?void 0:n.speechbrain_interface;if(i===void 0)return["# interface not specified in config.json"];const t=Ka(i);return t===void 0?["# interface in config.json invalid"]:[`from speechbrain.pretrained import ${i}
model = ${i}.from_hparams(
  "${e.id}"
)
model.${t}("file.wav")`]},Ja=e=>[`from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build("${e.id}")`],Xa=e=>{var n,o,s,d,u;const i=e.transformersInfo;if(!i)return["# ⚠️ Type of model unknown"];const t=e.tags.includes(Ce)?", trust_remote_code=True":"";let a;if(i.processor){const c=i.processor==="AutoTokenizer"?"tokenizer":i.processor==="AutoFeatureExtractor"?"extractor":"processor";a=["# Load model directly",`from transformers import ${i.processor}, ${i.auto_model}`,"",`${c} = ${i.processor}.from_pretrained("${e.id}"`+t+")",`model = ${i.auto_model}.from_pretrained("${e.id}"`+t+")"].join(`
`)}else a=["# Load model directly",`from transformers import ${i.auto_model}`,`model = ${i.auto_model}.from_pretrained("${e.id}"`+t+")"].join(`
`);if(e.pipeline_tag&&((n=z.transformers)!=null&&n.includes(e.pipeline_tag))){const c=["# Use a pipeline as a high-level helper","from transformers import pipeline",""];return e.tags.includes("conversational")&&((s=(o=e.config)==null?void 0:o.tokenizer_config)!=null&&s.chat_template)&&c.push("messages = [",'    {"role": "user", "content": "Who are you?"},',"]"),c.push(`pipe = pipeline("${e.pipeline_tag}", model="${e.id}"`+t+")"),e.tags.includes("conversational")&&((u=(d=e.config)==null?void 0:d.tokenizer_config)!=null&&u.chat_template)&&c.push("pipe(messages)"),[c.join(`
`),a]}return[a]},Ya=e=>{if(!e.pipeline_tag)return["// ⚠️ Unknown pipeline tag"];const i="@huggingface/transformers";return[`// npm i ${i}
import { pipeline } from '${i}';

// Allocate pipeline
const pipe = await pipeline('${e.pipeline_tag}', '${e.id}');`]},Ga=e=>{switch(e){case"CAUSAL_LM":return"CausalLM";case"SEQ_2_SEQ_LM":return"Seq2SeqLM";case"TOKEN_CLS":return"TokenClassification";case"SEQ_CLS":return"SequenceClassification";default:return}},Za=e=>{var n;const{base_model_name_or_path:i,task_type:t}=((n=e.config)==null?void 0:n.peft)??{},a=Ga(t);return a?i?[`from peft import PeftModel
from transformers import AutoModelFor${a}

base_model = AutoModelFor${a}.from_pretrained("${i}")
model = PeftModel.from_pretrained(base_model, "${e.id}")`]:["Base model is not found."]:["Task type is invalid."]},en=e=>[`from huggingface_hub import hf_hub_download
import fasttext

model = fasttext.load_model(hf_hub_download("${e.id}", "model.bin"))`],tn=e=>[`from huggingface_sb3 import load_from_hub
checkpoint = load_from_hub(
	repo_id="${e.id}",
	filename="{MODEL FILENAME}.zip",
)`],an=(e,i)=>{switch(e){case"ASR":return[`import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("${i.id}")

transcriptions = asr_model.transcribe(["file.wav"])`];default:return}},nn=e=>[`mlagents-load-from-hf --repo-id="${e.id}" --local-dir="./download: string[]s"`],on=()=>[`string modelName = "[Your model name here].sentis";
Model model = ModelLoader.Load(Application.streamingAssetsPath + "/" + modelName);
IWorker engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
// Please see provided C# file for more details
`],rn=e=>[`
# Load the model and infer image from text
import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

sana = SanaPipeline("configs/sana_config/1024ms/Sana_1600M_img1024.yaml")
sana.from_pretrained("hf://${e.id}")

image = sana(
    prompt='a cyberpunk cat with a neon sign that says "Sana"',
    height=1024,
    width=1024,
    guidance_scale=5.0,
    pag_guidance_scale=2.0,
    num_inference_steps=18,
) `],sn=e=>[`from Trainer_finetune import Model

model = Model.from_pretrained("${e.id}")`],ln=e=>[`from voicecraft import VoiceCraft

model = VoiceCraft.from_pretrained("${e.id}")`],cn=()=>[`import ChatTTS
import torchaudio

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["PUT YOUR TEXT HERE",]

wavs = chat.infer(texts, )

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)`],Le=e=>{const i=e.tags.find(n=>n.match(/^yolov\d+$/)),t=i?`YOLOv${i.slice(4)}`:"YOLOvXX";return[(i?"":`# Couldn't find a valid YOLO version tag.
# Replace XX with the correct version.
`)+`from ultralytics import ${t}

model = ${t}.from_pretrained("${e.id}")
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)`]},pn=e=>[`# Option 1: use with transformers

from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained("${e.id}", trust_remote_code=True)
`,`# Option 2: use with BiRefNet

# Install from https://github.com/ZhengPeng7/BiRefNet

from models.birefnet import BiRefNet
model = BiRefNet.from_pretrained("${e.id}")`],dn=e=>[`from swarmformer import SwarmFormerModel

model = SwarmFormerModel.from_pretrained("${e.id}")
`],un=e=>[`pip install huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --local-dir ${V(e.id)} ${e.id}`],mn=e=>[`from mlxim.model import create_model

model = create_model(${e.id})`],fn=e=>[`from model2vec import StaticModel

model = StaticModel.from_pretrained("${e.id}")`],hn=e=>{let i;return e.tags.includes("automatic-speech-recognition")&&(i=an("ASR",e)),i??["# tag did not correspond to a valid NeMo domain."]},gn=e=>[`from pxia import AutoModel

model = AutoModel.from_pretrained("${e.id}")`],bn=e=>[`from pythae.models import AutoModel

model = AutoModel.load_from_hf_hub("${e.id}")`],yn=e=>[`from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("${e.id}")

descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.`],wn=e=>[`from audiocraft.models import MAGNeT
	
model = MAGNeT.get_pretrained("${e.id}")

descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.`],vn=e=>[`from audiocraft.models import AudioGen
	
model = AudioGen.get_pretrained("${e.id}")
model.set_generation_params(duration=5)  # generate 5 seconds.
descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor']
wav = model.generate(descriptions)  # generates 3 samples.`];Object.entries({"adapter-transformers":{prettyLabel:"Adapters",repoName:"adapters",repoUrl:"https://github.com/Adapter-Hub/adapters",docsUrl:"https://huggingface.co/docs/hub/adapters",snippets:Li,filter:!0,countDownloads:'path:"adapter_config.json"'},allennlp:{prettyLabel:"AllenNLP",repoName:"AllenNLP",repoUrl:"https://github.com/allenai/allennlp",docsUrl:"https://huggingface.co/docs/hub/allennlp",snippets:$i,filter:!0},anemoi:{prettyLabel:"AnemoI",repoName:"AnemoI",repoUrl:"https://github.com/ecmwf/anemoi-inference",docsUrl:"https://anemoi-docs.readthedocs.io/en/latest/",filter:!1,countDownloads:'path_extension:"ckpt"',snippets:e=>[`from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.config import Configuration
# Create Configuration
config = Configuration(checkpoint = {"huggingface":{"repo_id":"${e.id}"}})
# Load Runner
runner = DefaultRunner(config)`]},araclip:{prettyLabel:"AraClip",repoName:"AraClip",repoUrl:"https://huggingface.co/Arabic-Clip/araclip",filter:!1,snippets:Ri},asteroid:{prettyLabel:"Asteroid",repoName:"Asteroid",repoUrl:"https://github.com/asteroid-team/asteroid",docsUrl:"https://huggingface.co/docs/hub/asteroid",snippets:Pi,filter:!0,countDownloads:'path:"pytorch_model.bin"'},audiocraft:{prettyLabel:"Audiocraft",repoName:"audiocraft",repoUrl:"https://github.com/facebookresearch/audiocraft",snippets:e=>e.tags.includes("musicgen")?yn(e):e.tags.includes("audiogen")?vn(e):e.tags.includes("magnet")?wn(e):["# Type of model unknown."],filter:!1,countDownloads:'path:"state_dict.bin"'},audioseal:{prettyLabel:"AudioSeal",repoName:"audioseal",repoUrl:"https://github.com/facebookresearch/audioseal",filter:!1,countDownloads:'path_extension:"pth"',snippets:ji},ben2:{prettyLabel:"BEN2",repoName:"BEN2",repoUrl:"https://github.com/PramaLLC/BEN2",snippets:Bi,filter:!1},bertopic:{prettyLabel:"BERTopic",repoName:"BERTopic",repoUrl:"https://github.com/MaartenGr/BERTopic",snippets:qi,filter:!0},big_vision:{prettyLabel:"Big Vision",repoName:"big_vision",repoUrl:"https://github.com/google-research/big_vision",filter:!1,countDownloads:'path_extension:"npz"'},birder:{prettyLabel:"Birder",repoName:"Birder",repoUrl:"https://gitlab.com/birder/birder",filter:!1,countDownloads:'path_extension:"pt"'},birefnet:{prettyLabel:"BiRefNet",repoName:"BiRefNet",repoUrl:"https://github.com/ZhengPeng7/BiRefNet",snippets:pn,filter:!1},bm25s:{prettyLabel:"BM25S",repoName:"bm25s",repoUrl:"https://github.com/xhluca/bm25s",snippets:Vi,filter:!1,countDownloads:'path:"params.index.json"'},champ:{prettyLabel:"Champ",repoName:"Champ",repoUrl:"https://github.com/fudan-generative-vision/champ",countDownloads:'path:"champ/motion_module.pth"'},chat_tts:{prettyLabel:"ChatTTS",repoName:"ChatTTS",repoUrl:"https://github.com/2noise/ChatTTS.git",snippets:cn,filter:!1,countDownloads:'path:"asset/GPT.pt"'},colpali:{prettyLabel:"ColPali",repoName:"ColPali",repoUrl:"https://github.com/ManuelFay/colpali",filter:!1,countDownloads:'path:"adapter_config.json"'},comet:{prettyLabel:"COMET",repoName:"COMET",repoUrl:"https://github.com/Unbabel/COMET/",countDownloads:'path:"hparams.yaml"'},cosmos:{prettyLabel:"Cosmos",repoName:"Cosmos",repoUrl:"https://github.com/NVIDIA/Cosmos",countDownloads:'path:"config.json" OR path_extension:"pt"'},"cxr-foundation":{prettyLabel:"CXR Foundation",repoName:"cxr-foundation",repoUrl:"https://github.com/google-health/cxr-foundation",snippets:Fi,filter:!1,countDownloads:'path:"precomputed_embeddings/embeddings.npz" OR path:"pax-elixr-b-text/saved_model.pb"'},deepforest:{prettyLabel:"DeepForest",repoName:"deepforest",docsUrl:"https://deepforest.readthedocs.io/en/latest/",repoUrl:"https://github.com/weecology/DeepForest"},"depth-anything-v2":{prettyLabel:"DepthAnythingV2",repoName:"Depth Anything V2",repoUrl:"https://github.com/DepthAnything/Depth-Anything-V2",snippets:zi,filter:!1,countDownloads:'path_extension:"pth"'},"depth-pro":{prettyLabel:"Depth Pro",repoName:"Depth Pro",repoUrl:"https://github.com/apple/ml-depth-pro",countDownloads:'path_extension:"pt"',snippets:Hi,filter:!1},"derm-foundation":{prettyLabel:"Derm Foundation",repoName:"derm-foundation",repoUrl:"https://github.com/google-health/derm-foundation",snippets:Wi,filter:!1,countDownloads:'path:"scin_dataset_precomputed_embeddings.npz" OR path:"saved_model.pb"'},diffree:{prettyLabel:"Diffree",repoName:"Diffree",repoUrl:"https://github.com/OpenGVLab/Diffree",filter:!1,countDownloads:'path:"diffree-step=000010999.ckpt"'},diffusers:{prettyLabel:"Diffusers",repoName:"🤗/diffusers",repoUrl:"https://github.com/huggingface/diffusers",docsUrl:"https://huggingface.co/docs/hub/diffusers",snippets:Yi,filter:!0},diffusionkit:{prettyLabel:"DiffusionKit",repoName:"DiffusionKit",repoUrl:"https://github.com/argmaxinc/DiffusionKit",snippets:Gi},doctr:{prettyLabel:"docTR",repoName:"doctr",repoUrl:"https://github.com/mindee/doctr"},cartesia_pytorch:{prettyLabel:"Cartesia Pytorch",repoName:"Cartesia Pytorch",repoUrl:"https://github.com/cartesia-ai/cartesia_pytorch",snippets:Zi},cartesia_mlx:{prettyLabel:"Cartesia MLX",repoName:"Cartesia MLX",repoUrl:"https://github.com/cartesia-ai/cartesia_mlx",snippets:ea},clipscope:{prettyLabel:"clipscope",repoName:"clipscope",repoUrl:"https://github.com/Lewington-pitsos/clipscope",filter:!1,countDownloads:'path_extension:"pt"'},cosyvoice:{prettyLabel:"CosyVoice",repoName:"CosyVoice",repoUrl:"https://github.com/FunAudioLLM/CosyVoice",filter:!1,countDownloads:'path_extension:"onnx" OR path_extension:"pt"'},cotracker:{prettyLabel:"CoTracker",repoName:"CoTracker",repoUrl:"https://github.com/facebookresearch/co-tracker",filter:!1,countDownloads:'path_extension:"pth"'},edsnlp:{prettyLabel:"EDS-NLP",repoName:"edsnlp",repoUrl:"https://github.com/aphp/edsnlp",docsUrl:"https://aphp.github.io/edsnlp/latest/",filter:!1,snippets:ta,countDownloads:'path_filename:"config" AND path_extension:"cfg"'},elm:{prettyLabel:"ELM",repoName:"elm",repoUrl:"https://github.com/slicex-ai/elm",filter:!1,countDownloads:'path_filename:"slicex_elm_config" AND path_extension:"json"'},espnet:{prettyLabel:"ESPnet",repoName:"ESPnet",repoUrl:"https://github.com/espnet/espnet",docsUrl:"https://huggingface.co/docs/hub/espnet",snippets:oa,filter:!0},fairseq:{prettyLabel:"Fairseq",repoName:"fairseq",repoUrl:"https://github.com/pytorch/fairseq",snippets:ra,filter:!0},fastai:{prettyLabel:"fastai",repoName:"fastai",repoUrl:"https://github.com/fastai/fastai",docsUrl:"https://huggingface.co/docs/hub/fastai",snippets:Pa,filter:!0},fasttext:{prettyLabel:"fastText",repoName:"fastText",repoUrl:"https://fasttext.cc/",snippets:en,filter:!0,countDownloads:'path_extension:"bin"'},flair:{prettyLabel:"Flair",repoName:"Flair",repoUrl:"https://github.com/flairNLP/flair",docsUrl:"https://huggingface.co/docs/hub/flair",snippets:sa,filter:!0,countDownloads:'path:"pytorch_model.bin"'},"gemma.cpp":{prettyLabel:"gemma.cpp",repoName:"gemma.cpp",repoUrl:"https://github.com/google/gemma.cpp",filter:!1,countDownloads:'path_extension:"sbs"'},gliner:{prettyLabel:"GLiNER",repoName:"GLiNER",repoUrl:"https://github.com/urchade/GLiNER",snippets:la,filter:!1,countDownloads:'path:"gliner_config.json"'},"glyph-byt5":{prettyLabel:"Glyph-ByT5",repoName:"Glyph-ByT5",repoUrl:"https://github.com/AIGText/Glyph-ByT5",filter:!1,countDownloads:'path:"checkpoints/byt5_model.pt"'},grok:{prettyLabel:"Grok",repoName:"Grok",repoUrl:"https://github.com/xai-org/grok-1",filter:!1,countDownloads:'path:"ckpt/tensor00000_000" OR path:"ckpt-0/tensor00000_000"'},hallo:{prettyLabel:"Hallo",repoName:"Hallo",repoUrl:"https://github.com/fudan-generative-vision/hallo",countDownloads:'path:"hallo/net.pth"'},hezar:{prettyLabel:"Hezar",repoName:"Hezar",repoUrl:"https://github.com/hezarai/hezar",docsUrl:"https://hezarai.github.io/hezar",countDownloads:'path:"model_config.yaml" OR path:"embedding/embedding_config.yaml"'},htrflow:{prettyLabel:"HTRflow",repoName:"HTRflow",repoUrl:"https://github.com/AI-Riksarkivet/htrflow",docsUrl:"https://ai-riksarkivet.github.io/htrflow",snippets:ca},"hunyuan-dit":{prettyLabel:"HunyuanDiT",repoName:"HunyuanDiT",repoUrl:"https://github.com/Tencent/HunyuanDiT",countDownloads:'path:"pytorch_model_ema.pt" OR path:"pytorch_model_distill.pt"'},"hunyuan3d-2":{prettyLabel:"Hunyuan3D-2",repoName:"Hunyuan3D-2",repoUrl:"https://github.com/Tencent/Hunyuan3D-2",countDownloads:'path_filename:"model_index" OR path_filename:"config"'},imstoucan:{prettyLabel:"IMS Toucan",repoName:"IMS-Toucan",repoUrl:"https://github.com/DigitalPhonetics/IMS-Toucan",countDownloads:'path:"embedding_gan.pt" OR path:"Vocoder.pt" OR path:"ToucanTTS.pt"'},keras:{prettyLabel:"Keras",repoName:"Keras",repoUrl:"https://github.com/keras-team/keras",docsUrl:"https://huggingface.co/docs/hub/keras",snippets:pa,filter:!0,countDownloads:'path:"config.json" OR path_extension:"keras"'},"tf-keras":{prettyLabel:"TF-Keras",repoName:"TF-Keras",repoUrl:"https://github.com/keras-team/tf-keras",docsUrl:"https://huggingface.co/docs/hub/tf-keras",snippets:ha,countDownloads:'path:"saved_model.pb"'},"keras-hub":{prettyLabel:"KerasHub",repoName:"KerasHub",repoUrl:"https://github.com/keras-team/keras-hub",docsUrl:"https://keras.io/keras_hub/",snippets:ma,filter:!0},k2:{prettyLabel:"K2",repoName:"k2",repoUrl:"https://github.com/k2-fsa/k2"},liveportrait:{prettyLabel:"LivePortrait",repoName:"LivePortrait",repoUrl:"https://github.com/KwaiVGI/LivePortrait",filter:!1,countDownloads:'path:"liveportrait/landmark.onnx"'},"llama-cpp-python":{prettyLabel:"llama-cpp-python",repoName:"llama-cpp-python",repoUrl:"https://github.com/abetlen/llama-cpp-python",snippets:fa},"mini-omni2":{prettyLabel:"Mini-Omni2",repoName:"Mini-Omni2",repoUrl:"https://github.com/gpt-omni/mini-omni2",countDownloads:'path:"model_config.yaml"'},mindspore:{prettyLabel:"MindSpore",repoName:"mindspore",repoUrl:"https://github.com/mindspore-ai/mindspore"},"mamba-ssm":{prettyLabel:"MambaSSM",repoName:"MambaSSM",repoUrl:"https://github.com/state-spaces/mamba",filter:!1,snippets:ga},"mars5-tts":{prettyLabel:"MARS5-TTS",repoName:"MARS5-TTS",repoUrl:"https://github.com/Camb-ai/MARS5-TTS",filter:!1,countDownloads:'path:"mars5_ar.safetensors"',snippets:ba},matanyone:{prettyLabel:"MatAnyone",repoName:"MatAnyone",repoUrl:"https://github.com/pq-yang/MatAnyone",snippets:ya,filter:!1},"mesh-anything":{prettyLabel:"MeshAnything",repoName:"MeshAnything",repoUrl:"https://github.com/buaacyw/MeshAnything",filter:!1,countDownloads:'path:"MeshAnything_350m.pth"',snippets:wa},merlin:{prettyLabel:"Merlin",repoName:"Merlin",repoUrl:"https://github.com/StanfordMIMI/Merlin",filter:!1,countDownloads:'path_extension:"pt"'},medvae:{prettyLabel:"MedVAE",repoName:"MedVAE",repoUrl:"https://github.com/StanfordMIMI/MedVAE",filter:!1,countDownloads:'path_extension:"ckpt"'},mitie:{prettyLabel:"MITIE",repoName:"MITIE",repoUrl:"https://github.com/mit-nlp/MITIE",countDownloads:'path_filename:"total_word_feature_extractor"'},"ml-agents":{prettyLabel:"ml-agents",repoName:"ml-agents",repoUrl:"https://github.com/Unity-Technologies/ml-agents",docsUrl:"https://huggingface.co/docs/hub/ml-agents",snippets:nn,filter:!0,countDownloads:'path_extension:"onnx"'},mlx:{prettyLabel:"MLX",repoName:"MLX",repoUrl:"https://github.com/ml-explore/mlx-examples/tree/main",snippets:un,filter:!0},"mlx-image":{prettyLabel:"mlx-image",repoName:"mlx-image",repoUrl:"https://github.com/riccardomusmeci/mlx-image",docsUrl:"https://huggingface.co/docs/hub/mlx-image",snippets:mn,filter:!1,countDownloads:'path:"model.safetensors"'},"mlc-llm":{prettyLabel:"MLC-LLM",repoName:"MLC-LLM",repoUrl:"https://github.com/mlc-ai/mlc-llm",docsUrl:"https://llm.mlc.ai/docs/",filter:!1,countDownloads:'path:"mlc-chat-config.json"'},model2vec:{prettyLabel:"Model2Vec",repoName:"model2vec",repoUrl:"https://github.com/MinishLab/model2vec",snippets:fn,filter:!1},moshi:{prettyLabel:"Moshi",repoName:"Moshi",repoUrl:"https://github.com/kyutai-labs/moshi",filter:!1,countDownloads:'path:"tokenizer-e351c8d8-checkpoint125.safetensors"'},nemo:{prettyLabel:"NeMo",repoName:"NeMo",repoUrl:"https://github.com/NVIDIA/NeMo",snippets:hn,filter:!0,countDownloads:'path_extension:"nemo" OR path:"model_config.yaml"'},"open-oasis":{prettyLabel:"open-oasis",repoName:"open-oasis",repoUrl:"https://github.com/etched-ai/open-oasis",countDownloads:'path:"oasis500m.safetensors"'},open_clip:{prettyLabel:"OpenCLIP",repoName:"OpenCLIP",repoUrl:"https://github.com/mlfoundations/open_clip",snippets:va,filter:!0,countDownloads:`path:"open_clip_model.safetensors"
			OR path:"model.safetensors"
			OR path:"open_clip_pytorch_model.bin"
			OR path:"pytorch_model.bin"`},"open-sora":{prettyLabel:"Open-Sora",repoName:"Open-Sora",repoUrl:"https://github.com/hpcaitech/Open-Sora",filter:!1,countDownloads:'path:"Open_Sora_v2.safetensors"'},paddlenlp:{prettyLabel:"paddlenlp",repoName:"PaddleNLP",repoUrl:"https://github.com/PaddlePaddle/PaddleNLP",docsUrl:"https://huggingface.co/docs/hub/paddlenlp",snippets:_a,filter:!0,countDownloads:'path:"model_config.json"'},peft:{prettyLabel:"PEFT",repoName:"PEFT",repoUrl:"https://github.com/huggingface/peft",snippets:Za,filter:!0,countDownloads:'path:"adapter_config.json"'},pxia:{prettyLabel:"pxia",repoName:"pxia",repoUrl:"https://github.com/not-lain/pxia",snippets:gn,filter:!1},"pyannote-audio":{prettyLabel:"pyannote.audio",repoName:"pyannote-audio",repoUrl:"https://github.com/pyannote/pyannote-audio",snippets:Aa,filter:!0},"py-feat":{prettyLabel:"Py-Feat",repoName:"Py-Feat",repoUrl:"https://github.com/cosanlab/py-feat",docsUrl:"https://py-feat.org/",filter:!1},pythae:{prettyLabel:"pythae",repoName:"pythae",repoUrl:"https://github.com/clementchadebec/benchmark_VAE",snippets:bn,filter:!1},recurrentgemma:{prettyLabel:"RecurrentGemma",repoName:"recurrentgemma",repoUrl:"https://github.com/google-deepmind/recurrentgemma",filter:!1,countDownloads:'path:"tokenizer.model"'},relik:{prettyLabel:"Relik",repoName:"Relik",repoUrl:"https://github.com/SapienzaNLP/relik",snippets:Sa,filter:!1},refiners:{prettyLabel:"Refiners",repoName:"Refiners",repoUrl:"https://github.com/finegrain-ai/refiners",docsUrl:"https://refine.rs/",filter:!1,countDownloads:'path:"model.safetensors"'},reverb:{prettyLabel:"Reverb",repoName:"Reverb",repoUrl:"https://github.com/revdotcom/reverb",filter:!1},saelens:{prettyLabel:"SAELens",repoName:"SAELens",repoUrl:"https://github.com/jbloomAus/SAELens",snippets:Oa,filter:!1},sam2:{prettyLabel:"sam2",repoName:"sam2",repoUrl:"https://github.com/facebookresearch/segment-anything-2",filter:!1,snippets:ja,countDownloads:'path_extension:"pt"'},"sample-factory":{prettyLabel:"sample-factory",repoName:"sample-factory",repoUrl:"https://github.com/alex-petrenko/sample-factory",docsUrl:"https://huggingface.co/docs/hub/sample-factory",snippets:Ba,filter:!0,countDownloads:'path:"cfg.json"'},sapiens:{prettyLabel:"sapiens",repoName:"sapiens",repoUrl:"https://github.com/facebookresearch/sapiens",filter:!1,countDownloads:'path_extension:"pt2" OR path_extension:"pth" OR path_extension:"onnx"'},"sentence-transformers":{prettyLabel:"sentence-transformers",repoName:"sentence-transformers",repoUrl:"https://github.com/UKPLab/sentence-transformers",docsUrl:"https://huggingface.co/docs/hub/sentence-transformers",snippets:Va,filter:!0},setfit:{prettyLabel:"setfit",repoName:"setfit",repoUrl:"https://github.com/huggingface/setfit",docsUrl:"https://huggingface.co/docs/hub/setfit",snippets:Fa,filter:!0},sklearn:{prettyLabel:"Scikit-learn",repoName:"Scikit-learn",repoUrl:"https://github.com/scikit-learn/scikit-learn",snippets:$a,filter:!0,countDownloads:'path:"sklearn_model.joblib"'},spacy:{prettyLabel:"spaCy",repoName:"spaCy",repoUrl:"https://github.com/explosion/spaCy",docsUrl:"https://huggingface.co/docs/hub/spacy",snippets:za,filter:!0,countDownloads:'path_extension:"whl"'},"span-marker":{prettyLabel:"SpanMarker",repoName:"SpanMarkerNER",repoUrl:"https://github.com/tomaarsen/SpanMarkerNER",docsUrl:"https://huggingface.co/docs/hub/span_marker",snippets:Ha,filter:!0},speechbrain:{prettyLabel:"speechbrain",repoName:"speechbrain",repoUrl:"https://github.com/speechbrain/speechbrain",docsUrl:"https://huggingface.co/docs/hub/speechbrain",snippets:Qa,filter:!0,countDownloads:'path:"hyperparams.yaml"'},"ssr-speech":{prettyLabel:"SSR-Speech",repoName:"SSR-Speech",repoUrl:"https://github.com/WangHelin1997/SSR-Speech",filter:!1,countDownloads:'path_extension:".pth"'},"stable-audio-tools":{prettyLabel:"Stable Audio Tools",repoName:"stable-audio-tools",repoUrl:"https://github.com/Stability-AI/stable-audio-tools.git",filter:!1,countDownloads:'path:"model.safetensors"',snippets:Ra},"diffusion-single-file":{prettyLabel:"Diffusion Single File",repoName:"diffusion-single-file",repoUrl:"https://github.com/comfyanonymous/ComfyUI",filter:!1,countDownloads:'path_extension:"safetensors"'},"seed-story":{prettyLabel:"SEED-Story",repoName:"SEED-Story",repoUrl:"https://github.com/TencentARC/SEED-Story",filter:!1,countDownloads:'path:"cvlm_llama2_tokenizer/tokenizer.model"',snippets:Ma},soloaudio:{prettyLabel:"SoloAudio",repoName:"SoloAudio",repoUrl:"https://github.com/WangHelin1997/SoloAudio",filter:!1,countDownloads:'path:"soloaudio_v2.pt"'},"stable-baselines3":{prettyLabel:"stable-baselines3",repoName:"stable-baselines3",repoUrl:"https://github.com/huggingface/huggingface_sb3",docsUrl:"https://huggingface.co/docs/hub/stable-baselines3",snippets:tn,filter:!0,countDownloads:'path_extension:"zip"'},stanza:{prettyLabel:"Stanza",repoName:"stanza",repoUrl:"https://github.com/stanfordnlp/stanza",docsUrl:"https://huggingface.co/docs/hub/stanza",snippets:Wa,filter:!0,countDownloads:'path:"models/default.zip"'},swarmformer:{prettyLabel:"SwarmFormer",repoName:"SwarmFormer",repoUrl:"https://github.com/takara-ai/SwarmFormer",snippets:dn,filter:!1},"f5-tts":{prettyLabel:"F5-TTS",repoName:"F5-TTS",repoUrl:"https://github.com/SWivid/F5-TTS",filter:!1,countDownloads:'path_extension:"safetensors" OR path_extension:"pt"'},genmo:{prettyLabel:"Genmo",repoName:"Genmo",repoUrl:"https://github.com/genmoai/models",filter:!1,countDownloads:'path:"vae_stats.json"'},tensorflowtts:{prettyLabel:"TensorFlowTTS",repoName:"TensorFlowTTS",repoUrl:"https://github.com/TensorSpeech/TensorFlowTTS",snippets:Ca},tabpfn:{prettyLabel:"TabPFN",repoName:"TabPFN",repoUrl:"https://github.com/PriorLabs/TabPFN"},terratorch:{prettyLabel:"TerraTorch",repoName:"TerraTorch",repoUrl:"https://github.com/IBM/terratorch",docsUrl:"https://ibm.github.io/terratorch/",filter:!1,countDownloads:'path_extension:"pt"',snippets:Ja},"tic-clip":{prettyLabel:"TiC-CLIP",repoName:"TiC-CLIP",repoUrl:"https://github.com/apple/ml-tic-clip",filter:!1,countDownloads:'path_extension:"pt" AND path_prefix:"checkpoints/"'},timesfm:{prettyLabel:"TimesFM",repoName:"timesfm",repoUrl:"https://github.com/google-research/timesfm",filter:!1,countDownloads:'path:"checkpoints/checkpoint_1100000/state/checkpoint"'},timm:{prettyLabel:"timm",repoName:"pytorch-image-models",repoUrl:"https://github.com/rwightman/pytorch-image-models",docsUrl:"https://huggingface.co/docs/hub/timm",snippets:Ua,filter:!0,countDownloads:'path:"pytorch_model.bin" OR path:"model.safetensors"'},transformers:{prettyLabel:"Transformers",repoName:"🤗/transformers",repoUrl:"https://github.com/huggingface/transformers",docsUrl:"https://huggingface.co/docs/hub/transformers",snippets:Xa,filter:!0},"transformers.js":{prettyLabel:"Transformers.js",repoName:"transformers.js",repoUrl:"https://github.com/huggingface/transformers.js",docsUrl:"https://huggingface.co/docs/hub/transformers-js",snippets:Ya,filter:!0},trellis:{prettyLabel:"Trellis",repoName:"Trellis",repoUrl:"https://github.com/microsoft/TRELLIS",countDownloads:'path_extension:"safetensors"'},ultralytics:{prettyLabel:"ultralytics",repoName:"ultralytics",repoUrl:"https://github.com/ultralytics/ultralytics",docsUrl:"https://github.com/ultralytics/ultralytics",filter:!1,countDownloads:'path_extension:"pt"',snippets:Le},"uni-3dar":{prettyLabel:"Uni-3DAR",repoName:"Uni-3DAR",repoUrl:"https://github.com/dptech-corp/Uni-3DAR",docsUrl:"https://github.com/dptech-corp/Uni-3DAR",countDownloads:'path_extension:"pt"'},"unity-sentis":{prettyLabel:"unity-sentis",repoName:"unity-sentis",repoUrl:"https://github.com/Unity-Technologies/sentis-samples",snippets:on,filter:!0,countDownloads:'path_extension:"sentis"'},sana:{prettyLabel:"Sana",repoName:"Sana",repoUrl:"https://github.com/NVlabs/Sana",countDownloads:'path_extension:"pth"',snippets:rn},"vfi-mamba":{prettyLabel:"VFIMamba",repoName:"VFIMamba",repoUrl:"https://github.com/MCG-NJU/VFIMamba",countDownloads:'path_extension:"pkl"',snippets:sn},voicecraft:{prettyLabel:"VoiceCraft",repoName:"VoiceCraft",repoUrl:"https://github.com/jasonppy/VoiceCraft",docsUrl:"https://github.com/jasonppy/VoiceCraft",snippets:ln},wham:{prettyLabel:"WHAM",repoName:"wham",repoUrl:"https://huggingface.co/microsoft/wham",docsUrl:"https://huggingface.co/microsoft/wham/blob/main/README.md",countDownloads:'path_extension:"ckpt"'},whisperkit:{prettyLabel:"WhisperKit",repoName:"WhisperKit",repoUrl:"https://github.com/argmaxinc/WhisperKit",docsUrl:"https://github.com/argmaxinc/WhisperKit?tab=readme-ov-file#homebrew",snippets:()=>[`# Install CLI with Homebrew on macOS device
brew install whisperkit-cli

# View all available inference options
whisperkit-cli transcribe --help
	
# Download and run inference using whisper base model
whisperkit-cli transcribe --audio-path /path/to/audio.mp3

# Or use your preferred model variant
whisperkit-cli transcribe --model "large-v3" --model-prefix "distil" --audio-path /path/to/audio.mp3 --verbose`],countDownloads:'path_filename:"model" AND path_extension:"mil" AND _exists_:"path_prefix"'},yolov10:{prettyLabel:"YOLOv10",repoName:"YOLOv10",repoUrl:"https://github.com/THU-MIG/yolov10",docsUrl:"https://github.com/THU-MIG/yolov10",countDownloads:'path_extension:"pt" OR path_extension:"safetensors"',snippets:Le},"3dtopia-xl":{prettyLabel:"3DTopia-XL",repoName:"3DTopia-XL",repoUrl:"https://github.com/3DTopia/3DTopia-XL",filter:!1,countDownloads:'path:"model_vae_fp16.pt"',snippets:e=>[`from threedtopia_xl.models import threedtopia_xl

model = threedtopia_xl.from_pretrained("${e.id}")
model.generate(cond="path/to/image.png")`]}}).filter(([e,i])=>i.filter).map(([e])=>e);var he;(function(e){e[e.F32=0]="F32",e[e.F16=1]="F16",e[e.Q4_0=2]="Q4_0",e[e.Q4_1=3]="Q4_1",e[e.Q5_0=6]="Q5_0",e[e.Q5_1=7]="Q5_1",e[e.Q8_0=8]="Q8_0",e[e.Q8_1=9]="Q8_1",e[e.Q2_K=10]="Q2_K",e[e.Q3_K=11]="Q3_K",e[e.Q4_K=12]="Q4_K",e[e.Q5_K=13]="Q5_K",e[e.Q6_K=14]="Q6_K",e[e.Q8_K=15]="Q8_K",e[e.IQ2_XXS=16]="IQ2_XXS",e[e.IQ2_XS=17]="IQ2_XS",e[e.IQ3_XXS=18]="IQ3_XXS",e[e.IQ1_S=19]="IQ1_S",e[e.IQ4_NL=20]="IQ4_NL",e[e.IQ3_S=21]="IQ3_S",e[e.IQ2_S=22]="IQ2_S",e[e.IQ4_XS=23]="IQ4_XS",e[e.I8=24]="I8",e[e.I16=25]="I16",e[e.I32=26]="I32",e[e.I64=27]="I64",e[e.F64=28]="F64",e[e.IQ1_M=29]="IQ1_M",e[e.BF16=30]="BF16"})(he||(he={}));const _n=Object.values(he).filter(e=>typeof e=="string");new RegExp(`(?<quant>${_n.join("|")})(_(?<sizeVariation>[A-Z]+))?`);const kn=["python","js","sh"];var r=Object.freeze({Text:"Text",NumericLiteral:"NumericLiteral",BooleanLiteral:"BooleanLiteral",NullLiteral:"NullLiteral",StringLiteral:"StringLiteral",Identifier:"Identifier",Equals:"Equals",OpenParen:"OpenParen",CloseParen:"CloseParen",OpenStatement:"OpenStatement",CloseStatement:"CloseStatement",OpenExpression:"OpenExpression",CloseExpression:"CloseExpression",OpenSquareBracket:"OpenSquareBracket",CloseSquareBracket:"CloseSquareBracket",OpenCurlyBracket:"OpenCurlyBracket",CloseCurlyBracket:"CloseCurlyBracket",Comma:"Comma",Dot:"Dot",Colon:"Colon",Pipe:"Pipe",CallOperator:"CallOperator",AdditiveBinaryOperator:"AdditiveBinaryOperator",MultiplicativeBinaryOperator:"MultiplicativeBinaryOperator",ComparisonBinaryOperator:"ComparisonBinaryOperator",UnaryOperator:"UnaryOperator",Set:"Set",If:"If",For:"For",In:"In",Is:"Is",NotIn:"NotIn",Else:"Else",EndIf:"EndIf",ElseIf:"ElseIf",EndFor:"EndFor",And:"And",Or:"Or",Not:"UnaryOperator",Macro:"Macro",EndMacro:"EndMacro"}),De=Object.freeze({set:r.Set,for:r.For,in:r.In,is:r.Is,if:r.If,else:r.Else,endif:r.EndIf,elif:r.ElseIf,endfor:r.EndFor,and:r.And,or:r.Or,not:r.Not,"not in":r.NotIn,macro:r.Macro,endmacro:r.EndMacro,true:r.BooleanLiteral,false:r.BooleanLiteral,none:r.NullLiteral,True:r.BooleanLiteral,False:r.BooleanLiteral,None:r.NullLiteral}),H=class{constructor(e,i){this.value=e,this.type=i}};function Ne(e){return/\w/.test(e)}function ge(e){return/[0-9]/.test(e)}var xn=[["{%",r.OpenStatement],["%}",r.CloseStatement],["{{",r.OpenExpression],["}}",r.CloseExpression],["(",r.OpenParen],[")",r.CloseParen],["{",r.OpenCurlyBracket],["}",r.CloseCurlyBracket],["[",r.OpenSquareBracket],["]",r.CloseSquareBracket],[",",r.Comma],[".",r.Dot],[":",r.Colon],["|",r.Pipe],["<=",r.ComparisonBinaryOperator],[">=",r.ComparisonBinaryOperator],["==",r.ComparisonBinaryOperator],["!=",r.ComparisonBinaryOperator],["<",r.ComparisonBinaryOperator],[">",r.ComparisonBinaryOperator],["+",r.AdditiveBinaryOperator],["-",r.AdditiveBinaryOperator],["*",r.MultiplicativeBinaryOperator],["/",r.MultiplicativeBinaryOperator],["%",r.MultiplicativeBinaryOperator],["=",r.Equals]],An=new Map([["n",`
`],["t","	"],["r","\r"],["b","\b"],["f","\f"],["v","\v"],["'","'"],['"','"'],["\\","\\"]]);function Sn(e,i={}){return e.endsWith(`
`)&&(e=e.slice(0,-1)),e=e.replace(/{#.*?#}/gs,"{##}"),i.lstrip_blocks&&(e=e.replace(/^[ \t]*({[#%])/gm,"$1")),i.trim_blocks&&(e=e.replace(/([#%]})\n/g,"$1")),e.replace(/{##}/g,"").replace(/-%}\s*/g,"%}").replace(/\s*{%-/g,"{%").replace(/-}}\s*/g,"}}").replace(/\s*{{-/g,"{{")}function In(e,i={}){var s,d,u;const t=[],a=Sn(e,i);let n=0;const o=c=>{let f="";for(;c(a[n]);){if(a[n]==="\\"){if(++n,n>=a.length)throw new SyntaxError("Unexpected end of input");const p=a[n++],y=An.get(p);if(y===void 0)throw new SyntaxError(`Unexpected escaped character: ${p}`);f+=y;continue}if(f+=a[n++],n>=a.length)throw new SyntaxError("Unexpected end of input")}return f};e:for(;n<a.length;){const c=(s=t.at(-1))==null?void 0:s.type;if(c===void 0||c===r.CloseStatement||c===r.CloseExpression){let p="";for(;n<a.length&&!(a[n]==="{"&&(a[n+1]==="%"||a[n+1]==="{"));)p+=a[n++];if(p.length>0){t.push(new H(p,r.Text));continue}}o(p=>/\s/.test(p));const f=a[n];if(f==="-"||f==="+"){const p=(d=t.at(-1))==null?void 0:d.type;if(p===r.Text||p===void 0)throw new SyntaxError(`Unexpected character: ${f}`);switch(p){case r.Identifier:case r.NumericLiteral:case r.BooleanLiteral:case r.NullLiteral:case r.StringLiteral:case r.CloseParen:case r.CloseSquareBracket:break;default:{++n;const y=o(ge);t.push(new H(`${f}${y}`,y.length>0?r.NumericLiteral:r.UnaryOperator));continue}}}for(const[p,y]of xn)if(a.slice(n,n+p.length)===p){t.push(new H(p,y)),n+=p.length;continue e}if(f==="'"||f==='"'){++n;const p=o(y=>y!==f);t.push(new H(p,r.StringLiteral)),++n;continue}if(ge(f)){const p=o(ge);t.push(new H(p,r.NumericLiteral));continue}if(Ne(f)){const p=o(Ne),y=Object.hasOwn(De,p)?De[p]:r.Identifier;y===r.In&&((u=t.at(-1))==null?void 0:u.type)===r.Not?(t.pop(),t.push(new H("not in",r.NotIn))):t.push(new H(p,y));continue}throw new SyntaxError(`Unexpected character: ${f}`)}return t}var Y=class{constructor(){b(this,"type","Statement")}},En=class extends Y{constructor(i){super();b(this,"type","Program");this.body=i}},$e=class extends Y{constructor(i,t,a){super();b(this,"type","If");this.test=i,this.body=t,this.alternate=a}},Tn=class extends Y{constructor(i,t,a,n){super();b(this,"type","For");this.loopvar=i,this.iterable=t,this.body=a,this.defaultBlock=n}},Cn=class extends Y{constructor(i,t){super();b(this,"type","Set");this.assignee=i,this.value=t}},Un=class extends Y{constructor(i,t,a){super();b(this,"type","Macro");this.name=i,this.args=t,this.body=a}},$=class extends Y{constructor(){super(...arguments);b(this,"type","Expression")}},On=class extends ${constructor(i,t,a){super();b(this,"type","MemberExpression");this.object=i,this.property=t,this.computed=a}},Mn=class extends ${constructor(i,t){super();b(this,"type","CallExpression");this.callee=i,this.args=t}},W=class extends ${constructor(i){super();b(this,"type","Identifier");this.value=i}},K=class extends ${constructor(i){super();b(this,"type","Literal");this.value=i}},Ln=class extends K{constructor(){super(...arguments);b(this,"type","NumericLiteral")}},Re=class extends K{constructor(){super(...arguments);b(this,"type","StringLiteral")}},Pe=class extends K{constructor(){super(...arguments);b(this,"type","BooleanLiteral")}},je=class extends K{constructor(){super(...arguments);b(this,"type","NullLiteral")}},Dn=class extends K{constructor(){super(...arguments);b(this,"type","ArrayLiteral")}},Be=class extends K{constructor(){super(...arguments);b(this,"type","TupleLiteral")}},Nn=class extends K{constructor(){super(...arguments);b(this,"type","ObjectLiteral")}},ie=class extends ${constructor(i,t,a){super();b(this,"type","BinaryExpression");this.operator=i,this.left=t,this.right=a}},$n=class extends ${constructor(i,t){super();b(this,"type","FilterExpression");this.operand=i,this.filter=t}},Rn=class extends ${constructor(i,t){super();b(this,"type","SelectExpression");this.iterable=i,this.test=t}},Pn=class extends ${constructor(i,t,a){super();b(this,"type","TestExpression");this.operand=i,this.negate=t,this.test=a}},jn=class extends ${constructor(i,t){super();b(this,"type","UnaryExpression");this.operator=i,this.argument=t}},Bn=class extends ${constructor(i=void 0,t=void 0,a=void 0){super();b(this,"type","SliceExpression");this.start=i,this.stop=t,this.step=a}},qn=class extends ${constructor(i,t){super();b(this,"type","KeywordArgumentExpression");this.key=i,this.value=t}};function Vn(e){const i=new En([]);let t=0;function a(l,m){const h=e[t++];if(!h||h.type!==l)throw new Error(`Parser Error: ${m}. ${h.type} !== ${l}.`);return h}function n(){switch(e[t].type){case r.Text:return d();case r.OpenStatement:return u();case r.OpenExpression:return c();default:throw new SyntaxError(`Unexpected token type: ${e[t].type}`)}}function o(...l){return t+l.length<=e.length&&l.some((m,h)=>m!==e[t+h].type)}function s(...l){return t+l.length<=e.length&&l.every((m,h)=>m===e[t+h].type)}function d(){return new Re(a(r.Text,"Expected text token").value)}function u(){a(r.OpenStatement,"Expected opening statement token");let l;switch(e[t].type){case r.Set:++t,l=f(),a(r.CloseStatement,"Expected closing statement token");break;case r.If:++t,l=p(),a(r.OpenStatement,"Expected {% token"),a(r.EndIf,"Expected endif token"),a(r.CloseStatement,"Expected %} token");break;case r.Macro:++t,l=y(),a(r.OpenStatement,"Expected {% token"),a(r.EndMacro,"Expected endmacro token"),a(r.CloseStatement,"Expected %} token");break;case r.For:++t,l=k(),a(r.OpenStatement,"Expected {% token"),a(r.EndFor,"Expected endfor token"),a(r.CloseStatement,"Expected %} token");break;default:throw new SyntaxError(`Unknown statement type: ${e[t].type}`)}return l}function c(){a(r.OpenExpression,"Expected opening expression token");const l=C();return a(r.CloseExpression,"Expected closing expression token"),l}function f(){const l=C();if(s(r.Equals)){++t;const m=f();return new Cn(l,m)}return l}function p(){var M,ut,mt,ft,ht,gt,bt,yt;const l=C();a(r.CloseStatement,"Expected closing statement token");const m=[],h=[];for(;!(((M=e[t])==null?void 0:M.type)===r.OpenStatement&&(((ut=e[t+1])==null?void 0:ut.type)===r.ElseIf||((mt=e[t+1])==null?void 0:mt.type)===r.Else||((ft=e[t+1])==null?void 0:ft.type)===r.EndIf));)m.push(n());if(((ht=e[t])==null?void 0:ht.type)===r.OpenStatement&&((gt=e[t+1])==null?void 0:gt.type)!==r.EndIf)if(++t,s(r.ElseIf))a(r.ElseIf,"Expected elseif token"),h.push(p());else for(a(r.Else,"Expected else token"),a(r.CloseStatement,"Expected closing statement token");!(((bt=e[t])==null?void 0:bt.type)===r.OpenStatement&&((yt=e[t+1])==null?void 0:yt.type)===r.EndIf);)h.push(n());return new $e(l,m,h)}function y(){const l=ee();if(l.type!=="Identifier")throw new SyntaxError("Expected identifier following macro statement");const m=lt();a(r.CloseStatement,"Expected closing statement token");const h=[];for(;o(r.OpenStatement,r.EndMacro);)h.push(n());return new Un(l,m,h)}function _(l=!1){const m=l?ee:C,h=[m()],M=s(r.Comma);for(;M&&(++t,h.push(m()),!!s(r.Comma)););return M?new Be(h):h[0]}function k(){const l=_(!0);if(!(l instanceof W||l instanceof Be))throw new SyntaxError(`Expected identifier/tuple for the loop variable, got ${l.type} instead`);a(r.In,"Expected `in` keyword following loop variable");const m=C();a(r.CloseStatement,"Expected closing statement token");const h=[];for(;o(r.OpenStatement,r.EndFor)&&o(r.OpenStatement,r.Else);)h.push(n());const M=[];if(s(r.OpenStatement,r.Else))for(++t,++t,a(r.CloseStatement,"Expected closing statement token");o(r.OpenStatement,r.EndFor);)M.push(n());return new Tn(l,m,h,M)}function C(){return T()}function T(){const l=q();if(s(r.If)){++t;const m=q();if(s(r.Else)){++t;const h=q();return new $e(m,[l],[h])}else return new Rn(l,m)}return l}function q(){let l=Ae();for(;s(r.Or);){const m=e[t];++t;const h=Ae();l=new ie(m,l,h)}return l}function Ae(){let l=ne();for(;s(r.And);){const m=e[t];++t;const h=ne();l=new ie(m,l,h)}return l}function ne(){let l;for(;s(r.Not);){const m=e[t];++t;const h=ne();l=new jn(m,h)}return l??Se()}function Se(){let l=J();for(;s(r.ComparisonBinaryOperator)||s(r.In)||s(r.NotIn);){const m=e[t];++t;const h=J();l=new ie(m,l,h)}return l}function J(){let l=pt();for(;s(r.AdditiveBinaryOperator);){const m=e[t];++t;const h=pt();l=new ie(m,l,h)}return l}function Ie(){const l=ct(ee());return s(r.OpenParen)?oe(l):l}function oe(l){let m=new Mn(l,lt());return m=ct(m),s(r.OpenParen)&&(m=oe(m)),m}function lt(){a(r.OpenParen,"Expected opening parenthesis for arguments list");const l=Ls();return a(r.CloseParen,"Expected closing parenthesis for arguments list"),l}function Ls(){const l=[];for(;!s(r.CloseParen);){let m=C();if(s(r.Equals)){if(++t,!(m instanceof W))throw new SyntaxError("Expected identifier for keyword argument");const h=C();m=new qn(m,h)}l.push(m),s(r.Comma)&&++t}return l}function Ds(){const l=[];let m=!1;for(;!s(r.CloseSquareBracket);)s(r.Colon)?(l.push(void 0),++t,m=!0):(l.push(C()),s(r.Colon)&&(++t,m=!0));if(l.length===0)throw new SyntaxError("Expected at least one argument for member/slice expression");if(m){if(l.length>3)throw new SyntaxError("Expected 0-3 arguments for slice expression");return new Bn(...l)}return l[0]}function ct(l){for(;s(r.Dot)||s(r.OpenSquareBracket);){const m=e[t];++t;let h;const M=m.type!==r.Dot;if(M)h=Ds(),a(r.CloseSquareBracket,"Expected closing square bracket");else if(h=ee(),h.type!=="Identifier")throw new SyntaxError("Expected identifier following dot operator");l=new On(l,h,M)}return l}function pt(){let l=dt();for(;s(r.MultiplicativeBinaryOperator);){const m=e[t];++t;const h=dt();l=new ie(m,l,h)}return l}function dt(){let l=Ns();for(;s(r.Is);){++t;const m=s(r.Not);m&&++t;let h=ee();if(h instanceof Pe?h=new W(h.value.toString()):h instanceof je&&(h=new W("none")),!(h instanceof W))throw new SyntaxError("Expected identifier for the test");l=new Pn(l,m,h)}return l}function Ns(){let l=Ie();for(;s(r.Pipe);){++t;let m=ee();if(!(m instanceof W))throw new SyntaxError("Expected identifier for the filter");s(r.OpenParen)&&(m=oe(m)),l=new $n(l,m)}return l}function ee(){const l=e[t];switch(l.type){case r.NumericLiteral:return++t,new Ln(Number(l.value));case r.StringLiteral:return++t,new Re(l.value);case r.BooleanLiteral:return++t,new Pe(l.value.toLowerCase()==="true");case r.NullLiteral:return++t,new je(null);case r.Identifier:return++t,new W(l.value);case r.OpenParen:{++t;const m=_();if(e[t].type!==r.CloseParen)throw new SyntaxError(`Expected closing parenthesis, got ${e[t].type} instead`);return++t,m}case r.OpenSquareBracket:{++t;const m=[];for(;!s(r.CloseSquareBracket);)m.push(C()),s(r.Comma)&&++t;return++t,new Dn(m)}case r.OpenCurlyBracket:{++t;const m=new Map;for(;!s(r.CloseCurlyBracket);){const h=C();a(r.Colon,"Expected colon between key and value in object literal");const M=C();m.set(h,M),s(r.Comma)&&++t}return++t,new Nn(m)}default:throw new SyntaxError(`Unexpected token: ${l.type}`)}}for(;t<e.length;)i.body.push(n());return i}function Fn(e,i,t=1){i===void 0&&(i=e,e=0);const a=[];for(let n=e;n<i;n+=t)a.push(n);return a}function qe(e,i,t,a=1){const n=Math.sign(a);n>=0?(i=(i??(i=0))<0?Math.max(e.length+i,0):Math.min(i,e.length),t=(t??(t=e.length))<0?Math.max(e.length+t,0):Math.min(t,e.length)):(i=(i??(i=e.length-1))<0?Math.max(e.length+i,-1):Math.min(i,e.length-1),t=(t??(t=-1))<-1?Math.max(e.length+t,-1):Math.min(t,e.length-1));const o=[];for(let s=i;n*s<n*t;s+=a)o.push(e[s]);return o}function Ve(e){return e.replace(/\b\w/g,i=>i.toUpperCase())}var j=class{constructor(e=void 0){b(this,"type","RuntimeValue");b(this,"value");b(this,"builtins",new Map);this.value=e}__bool__(){return new E(!!this.value)}},A=class extends j{constructor(){super(...arguments);b(this,"type","NumericValue")}},g=class extends j{constructor(){super(...arguments);b(this,"type","StringValue");b(this,"builtins",new Map([["upper",new U(()=>new g(this.value.toUpperCase()))],["lower",new U(()=>new g(this.value.toLowerCase()))],["strip",new U(()=>new g(this.value.trim()))],["title",new U(()=>new g(Ve(this.value)))],["length",new A(this.value.length)],["rstrip",new U(()=>new g(this.value.trimEnd()))],["lstrip",new U(()=>new g(this.value.trimStart()))],["split",new U(i=>{const t=i[0]??new D;if(!(t instanceof g||t instanceof D))throw new Error("sep argument must be a string or null");const a=i[1]??new A(-1);if(!(a instanceof A))throw new Error("maxsplit argument must be a number");let n=[];if(t instanceof D){const o=this.value.trimStart();for(const{0:s,index:d}of o.matchAll(/\S+/g)){if(a.value!==-1&&n.length>=a.value&&d!==void 0){n.push(s+o.slice(d+s.length));break}n.push(s)}}else{if(t.value==="")throw new Error("empty separator");n=this.value.split(t.value),a.value!==-1&&n.length>a.value&&n.push(n.splice(a.value).join(t.value))}return new I(n.map(o=>new g(o)))})]]))}},E=class extends j{constructor(){super(...arguments);b(this,"type","BooleanValue")}},L=class extends j{constructor(){super(...arguments);b(this,"type","ObjectValue");b(this,"builtins",new Map([["get",new U(([i,t])=>{if(!(i instanceof g))throw new Error(`Object key must be a string: got ${i.type}`);return this.value.get(i.value)??t??new D})],["items",new U(()=>new I(Array.from(this.value.entries()).map(([i,t])=>new I([new g(i),t]))))]]))}__bool__(){return new E(this.value.size>0)}},zn=class extends L{constructor(){super(...arguments);b(this,"type","KeywordArgumentsValue")}},I=class extends j{constructor(){super(...arguments);b(this,"type","ArrayValue");b(this,"builtins",new Map([["length",new A(this.value.length)]]))}__bool__(){return new E(this.value.length>0)}},Hn=class extends I{constructor(){super(...arguments);b(this,"type","TupleValue")}},U=class extends j{constructor(){super(...arguments);b(this,"type","FunctionValue")}},D=class extends j{constructor(){super(...arguments);b(this,"type","NullValue")}},O=class extends j{constructor(){super(...arguments);b(this,"type","UndefinedValue")}},ae=class{constructor(e){b(this,"variables",new Map([["namespace",new U(e=>{if(e.length===0)return new L(new Map);if(e.length!==1||!(e[0]instanceof L))throw new Error("`namespace` expects either zero arguments or a single object argument");return e[0]})]]));b(this,"tests",new Map([["boolean",e=>e.type==="BooleanValue"],["callable",e=>e instanceof U],["odd",e=>{if(e.type!=="NumericValue")throw new Error(`Cannot apply test "odd" to type: ${e.type}`);return e.value%2!==0}],["even",e=>{if(e.type!=="NumericValue")throw new Error(`Cannot apply test "even" to type: ${e.type}`);return e.value%2===0}],["false",e=>e.type==="BooleanValue"&&!e.value],["true",e=>e.type==="BooleanValue"&&e.value],["none",e=>e.type==="NullValue"],["string",e=>e.type==="StringValue"],["number",e=>e.type==="NumericValue"],["integer",e=>e.type==="NumericValue"&&Number.isInteger(e.value)],["iterable",e=>e.type==="ArrayValue"||e.type==="StringValue"],["mapping",e=>e.type==="ObjectValue"],["lower",e=>{const i=e.value;return e.type==="StringValue"&&i===i.toLowerCase()}],["upper",e=>{const i=e.value;return e.type==="StringValue"&&i===i.toUpperCase()}],["none",e=>e.type==="NullValue"],["defined",e=>e.type!=="UndefinedValue"],["undefined",e=>e.type==="UndefinedValue"],["equalto",(e,i)=>e.value===i.value],["eq",(e,i)=>e.value===i.value]]));this.parent=e}set(e,i){return this.declareVariable(e,se(i))}declareVariable(e,i){if(this.variables.has(e))throw new SyntaxError(`Variable already declared: ${e}`);return this.variables.set(e,i),i}setVariable(e,i){return this.variables.set(e,i),i}resolve(e){if(this.variables.has(e))return this;if(this.parent)return this.parent.resolve(e);throw new Error(`Unknown variable: ${e}`)}lookupVariable(e){try{return this.resolve(e).variables.get(e)??new O}catch{return new O}}},Wn=class{constructor(e){b(this,"global");this.global=e??new ae}run(e){return this.evaluate(e,this.global)}evaluateBinaryExpression(e,i){const t=this.evaluate(e.left,i);switch(e.operator.value){case"and":return t.__bool__().value?this.evaluate(e.right,i):t;case"or":return t.__bool__().value?t:this.evaluate(e.right,i)}const a=this.evaluate(e.right,i);switch(e.operator.value){case"==":return new E(t.value==a.value);case"!=":return new E(t.value!=a.value)}if(t instanceof O||a instanceof O)throw new Error("Cannot perform operation on undefined values");if(t instanceof D||a instanceof D)throw new Error("Cannot perform operation on null values");if(t instanceof A&&a instanceof A)switch(e.operator.value){case"+":return new A(t.value+a.value);case"-":return new A(t.value-a.value);case"*":return new A(t.value*a.value);case"/":return new A(t.value/a.value);case"%":return new A(t.value%a.value);case"<":return new E(t.value<a.value);case">":return new E(t.value>a.value);case">=":return new E(t.value>=a.value);case"<=":return new E(t.value<=a.value)}else if(t instanceof I&&a instanceof I)switch(e.operator.value){case"+":return new I(t.value.concat(a.value))}else if(a instanceof I){const n=a.value.find(o=>o.value===t.value)!==void 0;switch(e.operator.value){case"in":return new E(n);case"not in":return new E(!n)}}if(t instanceof g||a instanceof g)switch(e.operator.value){case"+":return new g(t.value.toString()+a.value.toString())}if(t instanceof g&&a instanceof g)switch(e.operator.value){case"in":return new E(a.value.includes(t.value));case"not in":return new E(!a.value.includes(t.value))}if(t instanceof g&&a instanceof L)switch(e.operator.value){case"in":return new E(a.value.has(t.value));case"not in":return new E(!a.value.has(t.value))}throw new SyntaxError(`Unknown operator "${e.operator.value}" between ${t.type} and ${a.type}`)}evaluateArguments(e,i){const t=[],a=new Map;for(const n of e)if(n.type==="KeywordArgumentExpression"){const o=n;a.set(o.key.value,this.evaluate(o.value,i))}else{if(a.size>0)throw new Error("Positional arguments must come before keyword arguments");t.push(this.evaluate(n,i))}return[t,a]}evaluateFilterExpression(e,i){const t=this.evaluate(e.operand,i);if(e.filter.type==="Identifier"){const a=e.filter;if(a.value==="tojson")return new g(le(t));if(t instanceof I)switch(a.value){case"list":return t;case"first":return t.value[0];case"last":return t.value[t.value.length-1];case"length":return new A(t.value.length);case"reverse":return new I(t.value.reverse());case"sort":return new I(t.value.sort((n,o)=>{if(n.type!==o.type)throw new Error(`Cannot compare different types: ${n.type} and ${o.type}`);switch(n.type){case"NumericValue":return n.value-o.value;case"StringValue":return n.value.localeCompare(o.value);default:throw new Error(`Cannot compare type: ${n.type}`)}}));case"join":return new g(t.value.map(n=>n.value).join(""));default:throw new Error(`Unknown ArrayValue filter: ${a.value}`)}else if(t instanceof g)switch(a.value){case"length":return new A(t.value.length);case"upper":return new g(t.value.toUpperCase());case"lower":return new g(t.value.toLowerCase());case"title":return new g(Ve(t.value));case"capitalize":return new g(t.value.charAt(0).toUpperCase()+t.value.slice(1));case"trim":return new g(t.value.trim());case"indent":return new g(t.value.split(`
`).map((n,o)=>o===0||n.length===0?n:"    "+n).join(`
`));case"join":case"string":return t;default:throw new Error(`Unknown StringValue filter: ${a.value}`)}else if(t instanceof A)switch(a.value){case"abs":return new A(Math.abs(t.value));default:throw new Error(`Unknown NumericValue filter: ${a.value}`)}else if(t instanceof L)switch(a.value){case"items":return new I(Array.from(t.value.entries()).map(([n,o])=>new I([new g(n),o])));case"length":return new A(t.value.size);default:throw new Error(`Unknown ObjectValue filter: ${a.value}`)}throw new Error(`Cannot apply filter "${a.value}" to type: ${t.type}`)}else if(e.filter.type==="CallExpression"){const a=e.filter;if(a.callee.type!=="Identifier")throw new Error(`Unknown filter: ${a.callee.type}`);const n=a.callee.value;if(n==="tojson"){const[,o]=this.evaluateArguments(a.args,i),s=o.get("indent")??new D;if(!(s instanceof A||s instanceof D))throw new Error("If set, indent must be a number");return new g(le(t,s.value))}else if(n==="join"){let o;if(t instanceof g)o=Array.from(t.value);else if(t instanceof I)o=t.value.map(c=>c.value);else throw new Error(`Cannot apply filter "${n}" to type: ${t.type}`);const[s,d]=this.evaluateArguments(a.args,i),u=s.at(0)??d.get("separator")??new g("");if(!(u instanceof g))throw new Error("separator must be a string");return new g(o.join(u.value))}if(t instanceof I){switch(n){case"selectattr":case"rejectattr":{const o=n==="selectattr";if(t.value.some(p=>!(p instanceof L)))throw new Error(`\`${n}\` can only be applied to array of objects`);if(a.args.some(p=>p.type!=="StringLiteral"))throw new Error(`arguments of \`${n}\` must be strings`);const[s,d,u]=a.args.map(p=>this.evaluate(p,i));let c;if(d){const p=i.tests.get(d.value);if(!p)throw new Error(`Unknown test: ${d.value}`);c=p}else c=(...p)=>p[0].__bool__().value;const f=t.value.filter(p=>{const y=p.value.get(s.value),_=y?c(y,u):!1;return o?_:!_});return new I(f)}case"map":{const[,o]=this.evaluateArguments(a.args,i);if(o.has("attribute")){const s=o.get("attribute");if(!(s instanceof g))throw new Error("attribute must be a string");const d=o.get("default"),u=t.value.map(c=>{if(!(c instanceof L))throw new Error("items in map must be an object");return c.value.get(s.value)??d??new O});return new I(u)}else throw new Error("`map` expressions without `attribute` set are not currently supported.")}}throw new Error(`Unknown ArrayValue filter: ${n}`)}else if(t instanceof g){switch(n){case"indent":{const[o,s]=this.evaluateArguments(a.args,i),d=o.at(0)??s.get("width")??new A(4);if(!(d instanceof A))throw new Error("width must be a number");const u=o.at(1)??s.get("first")??new E(!1),c=o.at(2)??s.get("blank")??new E(!1),f=t.value.split(`
`),p=" ".repeat(d.value),y=f.map((_,k)=>!u.value&&k===0||!c.value&&_.length===0?_:p+_);return new g(y.join(`
`))}}throw new Error(`Unknown StringValue filter: ${n}`)}else throw new Error(`Cannot apply filter "${n}" to type: ${t.type}`)}throw new Error(`Unknown filter: ${e.filter.type}`)}evaluateTestExpression(e,i){const t=this.evaluate(e.operand,i),a=i.tests.get(e.test.value);if(!a)throw new Error(`Unknown test: ${e.test.value}`);const n=a(t);return new E(e.negate?!n:n)}evaluateUnaryExpression(e,i){const t=this.evaluate(e.argument,i);switch(e.operator.value){case"not":return new E(!t.value);default:throw new SyntaxError(`Unknown operator: ${e.operator.value}`)}}evalProgram(e,i){return this.evaluateBlock(e.body,i)}evaluateBlock(e,i){let t="";for(const a of e){const n=this.evaluate(a,i);n.type!=="NullValue"&&n.type!=="UndefinedValue"&&(t+=n.value)}return new g(t)}evaluateIdentifier(e,i){return i.lookupVariable(e.value)}evaluateCallExpression(e,i){const[t,a]=this.evaluateArguments(e.args,i);a.size>0&&t.push(new zn(a));const n=this.evaluate(e.callee,i);if(n.type!=="FunctionValue")throw new Error(`Cannot call something that is not a function: got ${n.type}`);return n.value(t,i)}evaluateSliceExpression(e,i,t){if(!(e instanceof I||e instanceof g))throw new Error("Slice object must be an array or string");const a=this.evaluate(i.start,t),n=this.evaluate(i.stop,t),o=this.evaluate(i.step,t);if(!(a instanceof A||a instanceof O))throw new Error("Slice start must be numeric or undefined");if(!(n instanceof A||n instanceof O))throw new Error("Slice stop must be numeric or undefined");if(!(o instanceof A||o instanceof O))throw new Error("Slice step must be numeric or undefined");return e instanceof I?new I(qe(e.value,a.value,n.value,o.value)):new g(qe(Array.from(e.value),a.value,n.value,o.value).join(""))}evaluateMemberExpression(e,i){const t=this.evaluate(e.object,i);let a;if(e.computed){if(e.property.type==="SliceExpression")return this.evaluateSliceExpression(t,e.property,i);a=this.evaluate(e.property,i)}else a=new g(e.property.value);let n;if(t instanceof L){if(!(a instanceof g))throw new Error(`Cannot access property with non-string: got ${a.type}`);n=t.value.get(a.value)??t.builtins.get(a.value)}else if(t instanceof I||t instanceof g)if(a instanceof A)n=t.value.at(a.value),t instanceof g&&(n=new g(t.value.at(a.value)));else if(a instanceof g)n=t.builtins.get(a.value);else throw new Error(`Cannot access property with non-string/non-number: got ${a.type}`);else{if(!(a instanceof g))throw new Error(`Cannot access property with non-string: got ${a.type}`);n=t.builtins.get(a.value)}return n instanceof j?n:new O}evaluateSet(e,i){const t=this.evaluate(e.value,i);if(e.assignee.type==="Identifier"){const a=e.assignee.value;i.setVariable(a,t)}else if(e.assignee.type==="MemberExpression"){const a=e.assignee,n=this.evaluate(a.object,i);if(!(n instanceof L))throw new Error("Cannot assign to member of non-object");if(a.property.type!=="Identifier")throw new Error("Cannot assign to member with non-identifier property");n.value.set(a.property.value,t)}else throw new Error(`Invalid LHS inside assignment expression: ${JSON.stringify(e.assignee)}`);return new D}evaluateIf(e,i){const t=this.evaluate(e.test,i);return this.evaluateBlock(t.__bool__().value?e.body:e.alternate,i)}evaluateFor(e,i){const t=new ae(i);let a,n;if(e.iterable.type==="SelectExpression"){const c=e.iterable;n=this.evaluate(c.iterable,t),a=c.test}else n=this.evaluate(e.iterable,t);if(!(n instanceof I))throw new Error(`Expected iterable type in for loop: got ${n.type}`);const o=[],s=[];for(let c=0;c<n.value.length;++c){const f=new ae(t),p=n.value[c];let y;if(e.loopvar.type==="Identifier")y=_=>_.setVariable(e.loopvar.value,p);else if(e.loopvar.type==="TupleLiteral"){const _=e.loopvar;if(p.type!=="ArrayValue")throw new Error(`Cannot unpack non-iterable type: ${p.type}`);const k=p;if(_.value.length!==k.value.length)throw new Error(`Too ${_.value.length>k.value.length?"few":"many"} items to unpack`);y=C=>{for(let T=0;T<_.value.length;++T){if(_.value[T].type!=="Identifier")throw new Error(`Cannot unpack non-identifier type: ${_.value[T].type}`);C.setVariable(_.value[T].value,k.value[T])}}}else throw new Error(`Invalid loop variable(s): ${e.loopvar.type}`);a&&(y(f),!this.evaluate(a,f).__bool__().value)||(o.push(p),s.push(y))}let d="",u=!0;for(let c=0;c<o.length;++c){const f=new Map([["index",new A(c+1)],["index0",new A(c)],["revindex",new A(o.length-c)],["revindex0",new A(o.length-c-1)],["first",new E(c===0)],["last",new E(c===o.length-1)],["length",new A(o.length)],["previtem",c>0?o[c-1]:new O],["nextitem",c<o.length-1?o[c+1]:new O]]);t.setVariable("loop",new L(f)),s[c](t);const p=this.evaluateBlock(e.body,t);d+=p.value,u=!1}if(u){const c=this.evaluateBlock(e.defaultBlock,t);d+=c.value}return new g(d)}evaluateMacro(e,i){return i.setVariable(e.name.value,new U((t,a)=>{var s;const n=new ae(a);t=t.slice();let o;((s=t.at(-1))==null?void 0:s.type)==="KeywordArgumentsValue"&&(o=t.pop());for(let d=0;d<e.args.length;++d){const u=e.args[d],c=t[d];if(u.type==="Identifier"){const f=u;if(!c)throw new Error(`Missing positional argument: ${f.value}`);n.setVariable(f.value,c)}else if(u.type==="KeywordArgumentExpression"){const f=u,p=c??(o==null?void 0:o.value.get(f.key.value))??this.evaluate(f.value,n);n.setVariable(f.key.value,p)}else throw new Error(`Unknown argument type: ${u.type}`)}return this.evaluateBlock(e.body,n)})),new D}evaluate(e,i){if(e===void 0)return new O;switch(e.type){case"Program":return this.evalProgram(e,i);case"Set":return this.evaluateSet(e,i);case"If":return this.evaluateIf(e,i);case"For":return this.evaluateFor(e,i);case"Macro":return this.evaluateMacro(e,i);case"NumericLiteral":return new A(Number(e.value));case"StringLiteral":return new g(e.value);case"BooleanLiteral":return new E(e.value);case"NullLiteral":return new D(e.value);case"ArrayLiteral":return new I(e.value.map(t=>this.evaluate(t,i)));case"TupleLiteral":return new Hn(e.value.map(t=>this.evaluate(t,i)));case"ObjectLiteral":{const t=new Map;for(const[a,n]of e.value){const o=this.evaluate(a,i);if(!(o instanceof g))throw new Error(`Object keys must be strings: got ${o.type}`);t.set(o.value,this.evaluate(n,i))}return new L(t)}case"Identifier":return this.evaluateIdentifier(e,i);case"CallExpression":return this.evaluateCallExpression(e,i);case"MemberExpression":return this.evaluateMemberExpression(e,i);case"UnaryExpression":return this.evaluateUnaryExpression(e,i);case"BinaryExpression":return this.evaluateBinaryExpression(e,i);case"FilterExpression":return this.evaluateFilterExpression(e,i);case"TestExpression":return this.evaluateTestExpression(e,i);default:throw new SyntaxError(`Unknown node type: ${e.type}`)}}};function se(e){switch(typeof e){case"number":return new A(e);case"string":return new g(e);case"boolean":return new E(e);case"undefined":return new O;case"object":return e===null?new D:Array.isArray(e)?new I(e.map(se)):new L(new Map(Object.entries(e).map(([i,t])=>[i,se(t)])));case"function":return new U((i,t)=>{const a=e(...i.map(n=>n.value))??null;return se(a)});default:throw new Error(`Cannot convert to runtime value: ${e}`)}}function le(e,i,t){const a=t??0;switch(e.type){case"NullValue":case"UndefinedValue":return"null";case"NumericValue":case"StringValue":case"BooleanValue":return JSON.stringify(e.value);case"ArrayValue":case"ObjectValue":{const n=i?" ".repeat(i):"",o=`
`+n.repeat(a),s=o+n;if(e.type==="ArrayValue"){const d=e.value.map(u=>le(u,i,a+1));return i?`[${s}${d.join(`,${s}`)}${o}]`:`[${d.join(", ")}]`}else{const d=Array.from(e.value.entries()).map(([u,c])=>{const f=`"${u}": ${le(c,i,a+1)}`;return i?`${s}${f}`:f});return i?`{${d.join(",")}${o}}`:`{${d.join(", ")}}`}}default:throw new Error(`Cannot convert to JSON: ${e.type}`)}}var Kn=class{constructor(e){b(this,"parsed");const i=In(e,{lstrip_blocks:!0,trim_blocks:!0});this.parsed=Vn(i)}render(e){const i=new ae;if(i.set("false",!1),i.set("true",!0),i.set("raise_exception",n=>{throw new Error(n)}),i.set("range",Fn),e)for(const[n,o]of Object.entries(e))i.set(n,o);return new Wn(i).run(this.parsed).value}},Qn=Object.defineProperty,Fe=(e,i)=>{for(var t in i)Qn(e,t,{get:i[t],enumerable:!0})},be={};Fe(be,{audioClassification:()=>Mr,audioToAudio:()=>$r,automaticSpeechRecognition:()=>Lr,chatCompletion:()=>ls,chatCompletionStream:()=>cs,documentQuestionAnswering:()=>ps,featureExtraction:()=>Jr,fillMask:()=>Xr,imageClassification:()=>Pr,imageSegmentation:()=>jr,imageToImage:()=>Hr,imageToText:()=>Br,objectDetection:()=>qr,questionAnswering:()=>Yr,request:()=>x,sentenceSimilarity:()=>Gr,streamingRequest:()=>pe,summarization:()=>es,tableQuestionAnswering:()=>ts,tabularClassification:()=>ms,tabularRegression:()=>us,textClassification:()=>is,textGeneration:()=>as,textGenerationStream:()=>ns,textToImage:()=>Fr,textToSpeech:()=>Nr,textToVideo:()=>Qr,tokenClassification:()=>os,translation:()=>rs,visualQuestionAnswering:()=>ds,zeroShotClassification:()=>ss,zeroShotImageClassification:()=>Kr});var ze="https://huggingface.co",He="https://router.huggingface.co",Jn="https://api.us1.bfl.ai",Xn=()=>Jn,Yn=e=>e.args,Gn=e=>e.authMethod==="provider-key"?{"X-Key":`${e.accessToken}`}:{Authorization:`Bearer ${e.accessToken}`},Zn=e=>`${e.baseUrl}/v1/${e.model}`,eo={makeBaseUrl:Xn,makeBody:Yn,makeHeaders:Gn,makeUrl:Zn},to="https://api.cerebras.ai",io=()=>to,ao=e=>({...e.args,model:e.model}),no=e=>({Authorization:`Bearer ${e.accessToken}`}),oo=e=>`${e.baseUrl}/v1/chat/completions`,ro={makeBaseUrl:io,makeBody:ao,makeHeaders:no,makeUrl:oo},so="https://api.cohere.com",lo=()=>so,co=e=>({...e.args,model:e.model}),po=e=>({Authorization:`Bearer ${e.accessToken}`}),uo=e=>`${e.baseUrl}/compatibility/v1/chat/completions`,mo={makeBaseUrl:lo,makeBody:co,makeHeaders:po,makeUrl:uo},v=class extends TypeError{constructor(e){super(`Invalid inference output: ${e}. Use the 'request' method with the same parameters to do a custom call with no type checking.`),this.name="InferenceOutputError"}};function ce(e){return/^http(s?):/.test(e)||e.startsWith("/")}function We(e){return new Promise(i=>{setTimeout(()=>i(),e)})}var fo="https://fal.run",ho="https://queue.fal.run",go=e=>e==="text-to-video"?ho:fo,bo=e=>e.args,yo=e=>({Authorization:e.authMethod==="provider-key"?`Key ${e.accessToken}`:`Bearer ${e.accessToken}`}),wo=e=>{const i=`${e.baseUrl}/${e.model}`;return e.authMethod!=="provider-key"&&e.task==="text-to-video"?`${i}?_subdomain=queue`:i},vo={makeBaseUrl:go,makeBody:bo,makeHeaders:yo,makeUrl:wo};async function _o(e,i,t){if(!e.request_id)throw new v("No request ID found in the response");let n=e.status;const o=new URL(i),s=`${o.protocol}//${o.host}${o.host==="router.huggingface.co"?"/fal-ai":""}`,d=new URL(e.response_url).pathname,u=o.search,c=`${s}${d}/status${u}`,f=`${s}${d}${u}`;for(;n!=="COMPLETED";){await We(500);const _=await fetch(c,{headers:t});if(!_.ok)throw new v("Failed to fetch response status from fal-ai API");try{n=(await _.json()).status}catch{throw new v("Failed to parse status response from fal-ai API")}}const p=await fetch(f,{headers:t});let y;try{y=await p.json()}catch{throw new v("Failed to parse result response from fal-ai API")}if(typeof y=="object"&&y&&"video"in y&&typeof y.video=="object"&&y.video&&"url"in y.video&&typeof y.video.url=="string"&&ce(y.video.url))return await(await fetch(y.video.url)).blob();throw new v("Expected { video: { url: string } } result format, got instead: "+JSON.stringify(y))}var ko="https://api.fireworks.ai",xo=()=>ko,Ao=e=>({...e.args,...e.chatCompletion?{model:e.model}:void 0}),So=e=>({Authorization:`Bearer ${e.accessToken}`}),Io=e=>e.chatCompletion?`${e.baseUrl}/inference/v1/chat/completions`:`${e.baseUrl}/inference`,Eo={makeBaseUrl:xo,makeBody:Ao,makeHeaders:So,makeUrl:Io},To=()=>`${He}/hf-inference`,Co=e=>({...e.args,...e.chatCompletion?{model:e.model}:void 0}),Uo=e=>({Authorization:`Bearer ${e.accessToken}`}),Oo=e=>e.task&&["feature-extraction","sentence-similarity"].includes(e.task)?`${e.baseUrl}/pipeline/${e.task}/${e.model}`:e.chatCompletion?`${e.baseUrl}/models/${e.model}/v1/chat/completions`:`${e.baseUrl}/models/${e.model}`,Mo={makeBaseUrl:To,makeBody:Co,makeHeaders:Uo,makeUrl:Oo},Lo="https://api.hyperbolic.xyz",Do=()=>Lo,No=e=>({...e.args,...e.task==="text-to-image"?{model_name:e.model}:{model:e.model}}),$o=e=>({Authorization:`Bearer ${e.accessToken}`}),Ro=e=>e.task==="text-to-image"?`${e.baseUrl}/v1/images/generations`:`${e.baseUrl}/v1/chat/completions`,Po={makeBaseUrl:Do,makeBody:No,makeHeaders:$o,makeUrl:Ro},jo="https://api.studio.nebius.ai",Bo=()=>jo,qo=e=>({...e.args,model:e.model}),Vo=e=>({Authorization:`Bearer ${e.accessToken}`}),Fo=e=>e.task==="text-to-image"?`${e.baseUrl}/v1/images/generations`:e.chatCompletion?`${e.baseUrl}/v1/chat/completions`:e.task==="text-generation"?`${e.baseUrl}/v1/completions`:e.baseUrl,zo={makeBaseUrl:Bo,makeBody:qo,makeHeaders:Vo,makeUrl:Fo},Ho="https://api.novita.ai",Wo=()=>Ho,Ko=e=>({...e.args,...e.chatCompletion?{model:e.model}:void 0}),Qo=e=>({Authorization:`Bearer ${e.accessToken}`}),Jo=e=>e.chatCompletion?`${e.baseUrl}/v3/openai/chat/completions`:e.task==="text-generation"?`${e.baseUrl}/v3/openai/completions`:e.task==="text-to-video"?`${e.baseUrl}/v3/hf/${e.model}`:e.baseUrl,Xo={makeBaseUrl:Wo,makeBody:Ko,makeHeaders:Qo,makeUrl:Jo},Yo="https://api.replicate.com",Go=()=>Yo,Zo=e=>({input:e.args,version:e.model.includes(":")?e.model.split(":")[1]:void 0}),er=e=>({Authorization:`Bearer ${e.accessToken}`,Prefer:"wait"}),tr=e=>e.model.includes(":")?`${e.baseUrl}/v1/predictions`:`${e.baseUrl}/v1/models/${e.model}/predictions`,ir={makeBaseUrl:Go,makeBody:Zo,makeHeaders:er,makeUrl:tr},ar="https://api.sambanova.ai",nr=()=>ar,or=e=>({...e.args,...e.chatCompletion?{model:e.model}:void 0}),rr=e=>({Authorization:`Bearer ${e.accessToken}`}),sr=e=>e.chatCompletion?`${e.baseUrl}/v1/chat/completions`:e.baseUrl,lr={makeBaseUrl:nr,makeBody:or,makeHeaders:rr,makeUrl:sr},cr="https://api.together.xyz",pr=()=>cr,dr=e=>({...e.args,model:e.model}),ur=e=>({Authorization:`Bearer ${e.accessToken}`}),mr=e=>e.task==="text-to-image"?`${e.baseUrl}/v1/images/generations`:e.chatCompletion?`${e.baseUrl}/v1/chat/completions`:e.task==="text-generation"?`${e.baseUrl}/v1/completions`:e.baseUrl,fr={makeBaseUrl:pr,makeBody:dr,makeHeaders:ur,makeUrl:mr},hr="https://api.openai.com",gr=()=>hr,br=e=>{if(!e.chatCompletion)throw new Error("OpenAI only supports chat completions.");return{...e.args,model:e.model}},yr=e=>({Authorization:`Bearer ${e.accessToken}`}),wr=e=>{if(!e.chatCompletion)throw new Error("OpenAI only supports chat completions.");return`${e.baseUrl}/v1/chat/completions`},vr={makeBaseUrl:gr,makeBody:br,makeHeaders:yr,makeUrl:wr,clientSideRoutingOnly:!0},_r="@huggingface/inference",kr="3.6.2",Ke={"black-forest-labs":{},cerebras:{},cohere:{},"fal-ai":{},"fireworks-ai":{},"hf-inference":{},hyperbolic:{},nebius:{},novita:{},openai:{},replicate:{},sambanova:{},together:{}},Qe=new Map;async function xr(e,i,t={}){var s,d;if(e.provider==="hf-inference")return e.model;if(!t.task)throw new Error("task must be specified when using a third-party provider");const a=t.task==="text-generation"&&t.chatCompletion?"conversational":t.task;if((s=Ke[e.provider])!=null&&s[e.model])return Ke[e.provider][e.model];let n;if(Qe.has(e.model)?n=Qe.get(e.model):n=await((t==null?void 0:t.fetch)??fetch)(`${ze}/api/models/${e.model}?expand[]=inferenceProviderMapping`,{headers:(d=i.accessToken)!=null&&d.startsWith("hf_")?{Authorization:`Bearer ${i.accessToken}`}:{}}).then(u=>u.json()).then(u=>u.inferenceProviderMapping).catch(()=>null),!n)throw new Error(`We have not been able to find inference provider information for model ${e.model}.`);const o=n[e.provider];if(o){if(o.task!==a)throw new Error(`Model ${e.model} is not supported for task ${a} and provider ${e.provider}. Supported task: ${o.task}.`);return o.status==="staging"&&console.warn(`Model ${e.model} is in staging mode for provider ${e.provider}. Meant for test purposes only.`),o.providerId}throw new Error(`Model ${e.model} is not supported provider ${e.provider}.`)}var Ar=`${He}/{{PROVIDER}}`,ye=null,Je={"black-forest-labs":eo,cerebras:ro,cohere:mo,"fal-ai":vo,"fireworks-ai":Eo,"hf-inference":Mo,hyperbolic:Po,openai:vr,nebius:zo,novita:Xo,replicate:ir,sambanova:lr,together:fr};async function we(e,i){const{provider:t,model:a}=e,n=t??"hf-inference",o=Je[n],{task:s,chatCompletion:d}=i??{};if(e.endpointUrl&&n!=="hf-inference")throw new Error("Cannot use endpointUrl with a third-party provider.");if(a&&ce(a))throw new Error("Model URLs are no longer supported. Use endpointUrl instead.");if(!a&&!s)throw new Error("No model provided, and no task has been specified.");if(!o)throw new Error(`No provider config found for provider ${n}`);if(o.clientSideRoutingOnly&&!a)throw new Error(`Provider ${n} requires a model ID to be passed directly.`);const u=a??await Sr(s),c=o.clientSideRoutingOnly?Er(a,n):await xr({model:u,provider:n},e,{task:s,chatCompletion:d,fetch:i==null?void 0:i.fetch});return Xe(c,e,i)}function Xe(e,i,t){const{accessToken:a,endpointUrl:n,provider:o,model:s,...d}=i,u=o??"hf-inference",c=Je[u],{includeCredentials:f,task:p,chatCompletion:y,signal:_}=t??{},k=(()=>{if(c.clientSideRoutingOnly){if(a&&a.startsWith("hf_"))throw new Error(`Provider ${u} is closed-source and does not support HF tokens.`);return"provider-key"}return a?a.startsWith("hf_")?"hf-token":"provider-key":f==="include"?"credentials-include":"none"})(),C=n?y?n+"/v1/chat/completions":n:c.makeUrl({authMethod:k,baseUrl:k!=="provider-key"?Ar.replace("{{PROVIDER}}",u):c.makeBaseUrl(p),model:e,chatCompletion:y,task:p}),T="data"in i&&!!i.data,q=c.makeHeaders({accessToken:a,authMethod:k});T||(q["Content-Type"]="application/json");const ne=[`${_r}/${kr}`,typeof navigator<"u"?navigator.userAgent:void 0].filter(oe=>oe!==void 0).join(" ");q["User-Agent"]=ne;const Se=T?i.data:JSON.stringify(c.makeBody({args:d,model:e,task:p,chatCompletion:y}));let J;typeof f=="string"?J=f:f===!0&&(J="include");const Ie={headers:q,method:"POST",body:Se,...J?{credentials:J}:void 0,signal:_};return{url:C,info:Ie}}async function Sr(e){ye||(ye=await Ir());const i=ye[e];if(((i==null?void 0:i.models.length)??0)<=0)throw new Error(`No default model defined for task ${e}, please define the model explicitly.`);return i.models[0].id}async function Ir(){const e=await fetch(`${ze}/api/tasks`);if(!e.ok)throw new Error("Failed to load tasks definitions from Hugging Face Hub.");return await e.json()}function Er(e,i){if(!e.startsWith(`${i}/`))throw new Error(`Models from ${i} must be prefixed by "${i}/". Got "${e}".`);return e.slice(i.length+1)}async function x(e,i){var o;const{url:t,info:a}=await we(e,i),n=await((i==null?void 0:i.fetch)??fetch)(t,a);if((i==null?void 0:i.retry_on_error)!==!1&&n.status===503)return x(e,i);if(!n.ok){const s=n.headers.get("Content-Type");if(["application/json","application/problem+json"].some(u=>s==null?void 0:s.startsWith(u))){const u=await n.json();throw[400,422,404,500].includes(n.status)&&(i!=null&&i.chatCompletion)?new Error(`Server ${e.model} does not seem to support chat completion. Error: ${JSON.stringify(u.error)}`):u.error||u.detail?new Error(JSON.stringify(u.error??u.detail)):new Error(u)}const d=s!=null&&s.startsWith("text/plain;")?await n.text():void 0;throw new Error(d??"An error occurred while fetching the blob")}return(o=n.headers.get("Content-Type"))!=null&&o.startsWith("application/json")?await n.json():await n.blob()}function Tr(e){let i,t,a,n=!1;return function(s){i===void 0?(i=s,t=0,a=-1):i=Ur(i,s);const d=i.length;let u=0;for(;t<d;){n&&(i[t]===10&&(u=++t),n=!1);let c=-1;for(;t<d&&c===-1;++t)switch(i[t]){case 58:a===-1&&(a=t-u);break;case 13:n=!0;case 10:c=t;break}if(c===-1)break;e(i.subarray(u,c),a),u=t,a=-1}u===d?i=void 0:u!==0&&(i=i.subarray(u),t-=u)}}function Cr(e,i,t){let a=Ye();const n=new TextDecoder;return function(s,d){if(s.length===0)t==null||t(a),a=Ye();else if(d>0){const u=n.decode(s.subarray(0,d)),c=d+(s[d+1]===32?2:1),f=n.decode(s.subarray(c));switch(u){case"data":a.data=a.data?a.data+`
`+f:f;break;case"event":a.event=f;break;case"id":e(a.id=f);break;case"retry":const p=parseInt(f,10);isNaN(p)||i(a.retry=p);break}}}}function Ur(e,i){const t=new Uint8Array(e.length+i.length);return t.set(e),t.set(i,e.length),t}function Ye(){return{data:"",event:"",id:"",retry:void 0}}async function*pe(e,i){var c,f;const{url:t,info:a}=await we({...e,stream:!0},i),n=await((i==null?void 0:i.fetch)??fetch)(t,a);if((i==null?void 0:i.retry_on_error)!==!1&&n.status===503)return yield*pe(e,i);if(!n.ok){if((c=n.headers.get("Content-Type"))!=null&&c.startsWith("application/json")){const p=await n.json();if([400,422,404,500].includes(n.status)&&(i!=null&&i.chatCompletion))throw new Error(`Server ${e.model} does not seem to support chat completion. Error: ${p.error}`);if(typeof p.error=="string")throw new Error(p.error);if(p.error&&"message"in p.error&&typeof p.error.message=="string")throw new Error(p.error.message)}throw new Error(`Server response contains error: ${n.status}`)}if(!((f=n.headers.get("content-type"))!=null&&f.startsWith("text/event-stream")))throw new Error("Server does not support event stream content type, it returned "+n.headers.get("content-type"));if(!n.body)return;const o=n.body.getReader();let s=[];const u=Tr(Cr(()=>{},()=>{},p=>{s.push(p)}));try{for(;;){const{done:p,value:y}=await o.read();if(p)return;u(y);for(const _ of s)if(_.data.length>0){if(_.data==="[DONE]")return;const k=JSON.parse(_.data);if(typeof k=="object"&&k!==null&&"error"in k){const C=typeof k.error=="string"?k.error:typeof k.error=="object"&&k.error&&"message"in k.error&&typeof k.error.message=="string"?k.error.message:JSON.stringify(k.error);throw new Error("Error forwarded from backend: "+C)}yield k}s=[]}}finally{o.releaseLock()}}function Or(e,i){return Object.assign({},...i.map(t=>{if(e[t]!==void 0)return{[t]:e[t]}}))}function Ge(e,i){return e.includes(i)}function R(e,i){const t=Array.isArray(i)?i:[i],a=Object.keys(e).filter(n=>!Ge(t,n));return Or(e,a)}function ve(e){return"data"in e?e:{...R(e,"inputs"),data:e.inputs}}async function Mr(e,i){const t=ve(e),a=await x(t,{...i,task:"audio-classification"});if(!(Array.isArray(a)&&a.every(o=>typeof o.label=="string"&&typeof o.score=="number")))throw new v("Expected Array<{label: string, score: number}>");return a}function G(e){if(globalThis.Buffer)return globalThis.Buffer.from(e).toString("base64");{const i=[];return e.forEach(t=>{i.push(String.fromCharCode(t))}),globalThis.btoa(i.join(""))}}async function Lr(e,i){const t=await Dr(e),a=await x(t,{...i,task:"automatic-speech-recognition"});if(!(typeof(a==null?void 0:a.text)=="string"))throw new v("Expected {text: string}");return a}var Ze=["audio/mpeg","audio/mp4","audio/wav","audio/x-wav"];async function Dr(e){if(e.provider==="fal-ai"){const i="data"in e&&e.data instanceof Blob?e.data:"inputs"in e?e.inputs:void 0,t=i==null?void 0:i.type;if(!t)throw new Error("Unable to determine the input's content-type. Make sure your are passing a Blob when using provider fal-ai.");if(!Ze.includes(t))throw new Error(`Provider fal-ai does not support blob type ${t} - supported content types are: ${Ze.join(", ")}`);const a=G(new Uint8Array(await i.arrayBuffer()));return{..."data"in e?R(e,"data"):R(e,"inputs"),audio_url:`data:${t};base64,${a}`}}else return ve(e)}async function Nr(e,i){const t=e.provider==="replicate"?{...R(e,["inputs","parameters"]),...e.parameters,text:e.inputs}:e,a=await x(t,{...i,task:"text-to-speech"});if(a instanceof Blob)return a;if(a&&typeof a=="object"&&"output"in a){if(typeof a.output=="string")return await(await fetch(a.output)).blob();if(Array.isArray(a.output))return await(await fetch(a.output[0])).blob()}throw new v("Expected Blob or object with output")}async function $r(e,i){const t=ve(e),a=await x(t,{...i,task:"audio-to-audio"});return Rr(a)}function Rr(e){if(!Array.isArray(e))throw new v("Expected Array");if(!e.every(i=>typeof i=="object"&&i&&"label"in i&&typeof i.label=="string"&&"content-type"in i&&typeof i["content-type"]=="string"&&"blob"in i&&typeof i.blob=="string"))throw new v("Expected Array<{label: string, audio: Blob}>");return e}function de(e){return"data"in e?e:{...R(e,"inputs"),data:e.inputs}}async function Pr(e,i){const t=de(e),a=await x(t,{...i,task:"image-classification"});if(!(Array.isArray(a)&&a.every(o=>typeof o.label=="string"&&typeof o.score=="number")))throw new v("Expected Array<{label: string, score: number}>");return a}async function jr(e,i){const t=de(e),a=await x(t,{...i,task:"image-segmentation"});if(!(Array.isArray(a)&&a.every(o=>typeof o.label=="string"&&typeof o.mask=="string"&&typeof o.score=="number")))throw new v("Expected Array<{label: string, mask: string, score: number}>");return a}async function Br(e,i){var n;const t=de(e),a=(n=await x(t,{...i,task:"image-to-text"}))==null?void 0:n[0];if(typeof(a==null?void 0:a.generated_text)!="string")throw new v("Expected {generated_text: string}");return a}async function qr(e,i){const t=de(e),a=await x(t,{...i,task:"object-detection"});if(!(Array.isArray(a)&&a.every(o=>typeof o.label=="string"&&typeof o.score=="number"&&typeof o.box.xmin=="number"&&typeof o.box.ymin=="number"&&typeof o.box.xmax=="number"&&typeof o.box.ymax=="number")))throw new v("Expected Array<{label:string; score:number; box:{xmin:number; ymin:number; xmax:number; ymax:number}}>");return a}function Vr(e){switch(e){case"fal-ai":return{sync_mode:!0};case"nebius":return{response_format:"b64_json"};case"replicate":return;case"together":return{response_format:"base64"};default:return}}async function Fr(e,i){const t=!e.provider||e.provider==="hf-inference"||e.provider==="sambanova"?e:{...R(e,["inputs","parameters"]),...e.parameters,...Vr(e.provider),prompt:e.inputs},a=await x(t,{...i,task:"text-to-image"});if(a&&typeof a=="object"){if(e.provider==="black-forest-labs"&&"polling_url"in a&&typeof a.polling_url=="string")return await zr(a.polling_url,i==null?void 0:i.outputType);if(e.provider==="fal-ai"&&"images"in a&&Array.isArray(a.images)&&a.images[0].url)return(i==null?void 0:i.outputType)==="url"?a.images[0].url:await(await fetch(a.images[0].url)).blob();if(e.provider==="hyperbolic"&&"images"in a&&Array.isArray(a.images)&&a.images[0]&&typeof a.images[0].image=="string")return(i==null?void 0:i.outputType)==="url"?`data:image/jpeg;base64,${a.images[0].image}`:await(await fetch(`data:image/jpeg;base64,${a.images[0].image}`)).blob();if("data"in a&&Array.isArray(a.data)&&a.data[0].b64_json){const o=a.data[0].b64_json;return(i==null?void 0:i.outputType)==="url"?`data:image/jpeg;base64,${o}`:await(await fetch(`data:image/jpeg;base64,${o}`)).blob()}if("output"in a&&Array.isArray(a.output))return(i==null?void 0:i.outputType)==="url"?a.output[0]:await(await fetch(a.output[0])).blob()}if(!(a&&a instanceof Blob))throw new v("Expected Blob");return(i==null?void 0:i.outputType)==="url"?`data:image/jpeg;base64,${await a.arrayBuffer().then(s=>Buffer.from(s).toString("base64"))}`:a}async function zr(e,i){const t=new URL(e);for(let a=0;a<5;a++){await We(1e3),console.debug(`Polling Black Forest Labs API for the result... ${a+1}/5`),t.searchParams.set("attempt",a.toString(10));const n=await fetch(t,{headers:{"Content-Type":"application/json"}});if(!n.ok)throw new v("Failed to fetch result from black forest labs API");const o=await n.json();if(typeof o=="object"&&o&&"status"in o&&typeof o.status=="string"&&o.status==="Ready"&&"result"in o&&typeof o.result=="object"&&o.result&&"sample"in o.result&&typeof o.result.sample=="string")return i==="url"?o.result.sample:await(await fetch(o.result.sample)).blob()}throw new v("Failed to fetch result from black forest labs API")}async function Hr(e,i){let t;e.parameters?t={...e,inputs:G(new Uint8Array(e.inputs instanceof ArrayBuffer?e.inputs:await e.inputs.arrayBuffer()))}:t={accessToken:e.accessToken,model:e.model,data:e.inputs};const a=await x(t,{...i,task:"image-to-image"});if(!(a&&a instanceof Blob))throw new v("Expected Blob");return a}async function Wr(e){return e.inputs instanceof Blob?{...e,inputs:{image:G(new Uint8Array(await e.inputs.arrayBuffer()))}}:{...e,inputs:{image:G(new Uint8Array(e.inputs.image instanceof ArrayBuffer?e.inputs.image:await e.inputs.image.arrayBuffer()))}}}async function Kr(e,i){const t=await Wr(e),a=await x(t,{...i,task:"zero-shot-image-classification"});if(!(Array.isArray(a)&&a.every(o=>typeof o.label=="string"&&typeof o.score=="number")))throw new v("Expected Array<{label: string, score: number}>");return a}var et=["fal-ai","novita","replicate"];async function Qr(e,i){if(!e.provider||!Ge(et,e.provider))throw new Error(`textToVideo inference is only supported for the following providers: ${et.join(", ")}`);const t=e.provider==="fal-ai"||e.provider==="replicate"||e.provider==="novita"?{...R(e,["inputs","parameters"]),...e.parameters,prompt:e.inputs}:e,a=await x(t,{...i,task:"text-to-video"});if(e.provider==="fal-ai"){const{url:n,info:o}=await we(e,{...i,task:"text-to-video"});return await _o(a,n,o.headers)}else if(e.provider==="novita"){if(!(typeof a=="object"&&!!a&&"video"in a&&typeof a.video=="object"&&!!a.video&&"video_url"in a.video&&typeof a.video.video_url=="string"&&ce(a.video.video_url)))throw new v("Expected { video: { video_url: string } }");return await(await fetch(a.video.video_url)).blob()}else{if(!(typeof a=="object"&&!!a&&"output"in a&&typeof a.output=="string"&&ce(a.output)))throw new v("Expected { output: string }");return await(await fetch(a.output)).blob()}}async function Jr(e,i){const t=await x(e,{...i,task:"feature-extraction"});let a=!0;const n=(o,s,d=0)=>d>s?!1:o.every(u=>Array.isArray(u))?o.every(u=>n(u,s,d+1)):o.every(u=>typeof u=="number");if(a=Array.isArray(t)&&n(t,3,0),!a)throw new v("Expected Array<number[][][] | number[][] | number[] | number>");return t}async function Xr(e,i){const t=await x(e,{...i,task:"fill-mask"});if(!(Array.isArray(t)&&t.every(n=>typeof n.score=="number"&&typeof n.sequence=="string"&&typeof n.token=="number"&&typeof n.token_str=="string")))throw new v("Expected Array<{score: number, sequence: string, token: number, token_str: string}>");return t}async function Yr(e,i){const t=await x(e,{...i,task:"question-answering"});if(!(Array.isArray(t)?t.every(n=>typeof n=="object"&&!!n&&typeof n.answer=="string"&&typeof n.end=="number"&&typeof n.score=="number"&&typeof n.start=="number"):typeof t=="object"&&!!t&&typeof t.answer=="string"&&typeof t.end=="number"&&typeof t.score=="number"&&typeof t.start=="number"))throw new v("Expected Array<{answer: string, end: number, score: number, start: number}>");return Array.isArray(t)?t[0]:t}async function Gr(e,i){const t=await x(Zr(e),{...i,task:"sentence-similarity"});if(!(Array.isArray(t)&&t.every(n=>typeof n=="number")))throw new v("Expected number[]");return t}function Zr(e){return{...R(e,["inputs","parameters"]),inputs:{...R(e.inputs,"sourceSentence")},parameters:{source_sentence:e.inputs.sourceSentence,...e.parameters}}}async function es(e,i){const t=await x(e,{...i,task:"summarization"});if(!(Array.isArray(t)&&t.every(n=>typeof(n==null?void 0:n.summary_text)=="string")))throw new v("Expected Array<{summary_text: string}>");return t==null?void 0:t[0]}async function ts(e,i){const t=await x(e,{...i,task:"table-question-answering"});if(!(Array.isArray(t)?t.every(n=>tt(n)):tt(t)))throw new v("Expected {aggregator: string, answer: string, cells: string[], coordinates: number[][]}");return Array.isArray(t)?t[0]:t}function tt(e){return typeof e=="object"&&!!e&&"aggregator"in e&&typeof e.aggregator=="string"&&"answer"in e&&typeof e.answer=="string"&&"cells"in e&&Array.isArray(e.cells)&&e.cells.every(i=>typeof i=="string")&&"coordinates"in e&&Array.isArray(e.coordinates)&&e.coordinates.every(i=>Array.isArray(i)&&i.every(t=>typeof t=="number"))}async function is(e,i){var n;const t=(n=await x(e,{...i,task:"text-classification"}))==null?void 0:n[0];if(!(Array.isArray(t)&&t.every(o=>typeof(o==null?void 0:o.label)=="string"&&typeof o.score=="number")))throw new v("Expected Array<{label: string, score: number}>");return t}function ue(e){return Array.isArray(e)?e:[e]}async function as(e,i){if(e.provider==="together"){e.prompt=e.inputs;const t=await x(e,{...i,task:"text-generation"});if(!(typeof t=="object"&&"choices"in t&&Array.isArray(t==null?void 0:t.choices)&&typeof(t==null?void 0:t.model)=="string"))throw new v("Expected ChatCompletionOutput");return{generated_text:t.choices[0].text}}else if(e.provider==="hyperbolic"){const t={messages:[{content:e.inputs,role:"user"}],...e.parameters?{max_tokens:e.parameters.max_new_tokens,...R(e.parameters,"max_new_tokens")}:void 0,...R(e,["inputs","parameters"])},a=await x(t,{...i,task:"text-generation"});if(!(typeof a=="object"&&"choices"in a&&Array.isArray(a==null?void 0:a.choices)&&typeof(a==null?void 0:a.model)=="string"))throw new v("Expected ChatCompletionOutput");return{generated_text:a.choices[0].message.content}}else{const t=ue(await x(e,{...i,task:"text-generation"}));if(!(Array.isArray(t)&&t.every(n=>"generated_text"in n&&typeof(n==null?void 0:n.generated_text)=="string")))throw new v("Expected Array<{generated_text: string}>");return t==null?void 0:t[0]}}async function*ns(e,i){yield*pe(e,{...i,task:"text-generation"})}async function os(e,i){const t=ue(await x(e,{...i,task:"token-classification"}));if(!(Array.isArray(t)&&t.every(n=>typeof n.end=="number"&&typeof n.entity_group=="string"&&typeof n.score=="number"&&typeof n.start=="number"&&typeof n.word=="string")))throw new v("Expected Array<{end: number, entity_group: string, score: number, start: number, word: string}>");return t}async function rs(e,i){const t=await x(e,{...i,task:"translation"});if(!(Array.isArray(t)&&t.every(n=>typeof(n==null?void 0:n.translation_text)=="string")))throw new v("Expected type Array<{translation_text: string}>");return(t==null?void 0:t.length)===1?t==null?void 0:t[0]:t}async function ss(e,i){const t=ue(await x(e,{...i,task:"zero-shot-classification"}));if(!(Array.isArray(t)&&t.every(n=>Array.isArray(n.labels)&&n.labels.every(o=>typeof o=="string")&&Array.isArray(n.scores)&&n.scores.every(o=>typeof o=="number")&&typeof n.sequence=="string")))throw new v("Expected Array<{labels: string[], scores: number[], sequence: string}>");return t}async function ls(e,i){const t=await x(e,{...i,task:"text-generation",chatCompletion:!0});if(!(typeof t=="object"&&Array.isArray(t==null?void 0:t.choices)&&typeof(t==null?void 0:t.created)=="number"&&typeof(t==null?void 0:t.id)=="string"&&typeof(t==null?void 0:t.model)=="string"&&(t.system_fingerprint===void 0||t.system_fingerprint===null||typeof t.system_fingerprint=="string")&&typeof(t==null?void 0:t.usage)=="object"))throw new v("Expected ChatCompletionOutput");return t}async function*cs(e,i){yield*pe(e,{...i,task:"text-generation",chatCompletion:!0})}async function ps(e,i){const t={...e,inputs:{question:e.inputs.question,image:G(new Uint8Array(await e.inputs.image.arrayBuffer()))}},a=ue(await x(t,{...i,task:"document-question-answering"}));if(!(Array.isArray(a)&&a.every(o=>typeof o=="object"&&!!o&&typeof(o==null?void 0:o.answer)=="string"&&(typeof o.end=="number"||typeof o.end>"u")&&(typeof o.score=="number"||typeof o.score>"u")&&(typeof o.start=="number"||typeof o.start>"u"))))throw new v("Expected Array<{answer: string, end?: number, score?: number, start?: number}>");return a[0]}async function ds(e,i){const t={...e,inputs:{question:e.inputs.question,image:G(new Uint8Array(await e.inputs.image.arrayBuffer()))}},a=await x(t,{...i,task:"visual-question-answering"});if(!(Array.isArray(a)&&a.every(o=>typeof o=="object"&&!!o&&typeof(o==null?void 0:o.answer)=="string"&&typeof o.score=="number")))throw new v("Expected Array<{answer: string, score: number}>");return a[0]}async function us(e,i){const t=await x(e,{...i,task:"tabular-regression"});if(!(Array.isArray(t)&&t.every(n=>typeof n=="number")))throw new v("Expected number[]");return t}async function ms(e,i){const t=await x(e,{...i,task:"tabular-classification"});if(!(Array.isArray(t)&&t.every(n=>typeof n=="number")))throw new v("Expected number[]");return t}var fs=class{constructor(e="",i={}){b(this,"accessToken");b(this,"defaultOptions");this.accessToken=e,this.defaultOptions=i;for(const[t,a]of Object.entries(be))Object.defineProperty(this,t,{enumerable:!1,value:(n,o)=>a({...n,accessToken:e},{...i,...o})})}endpoint(e){return new hs(e,this.accessToken,this.defaultOptions)}},hs=class{constructor(e,i="",t={}){for(const[a,n]of Object.entries(be))Object.defineProperty(this,a,{enumerable:!1,value:(o,s)=>n({...o,accessToken:i,endpointUrl:e},{...t,...s})})}},gs={};Fe(gs,{getInferenceSnippets:()=>Es});var it={js:{fetch:{basic:`async function query(data) {
	const response = await fetch(
		"{{ fullUrl }}",
		{
			headers: {
				Authorization: "{{ authorizationHeader }}",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({ inputs: {{ providerInputs.asObj.inputs }} }).then((response) => {
    console.log(JSON.stringify(response));
});`,basicAudio:`async function query(data) {
	const response = await fetch(
		"{{ fullUrl }}",
		{
			headers: {
				Authorization: "{{ authorizationHeader }}",
				"Content-Type": "audio/flac"
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({ inputs: {{ providerInputs.asObj.inputs }} }).then((response) => {
    console.log(JSON.stringify(response));
});`,basicImage:`async function query(data) {
	const response = await fetch(
		"{{ fullUrl }}",
		{
			headers: {
				Authorization: "{{ authorizationHeader }}",
				"Content-Type": "image/jpeg"
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}

query({ inputs: {{ providerInputs.asObj.inputs }} }).then((response) => {
    console.log(JSON.stringify(response));
});`,textToAudio:`{% if model.library_name == "transformers" %}
async function query(data) {
	const response = await fetch(
		"{{ fullUrl }}",
		{
			headers: {
				Authorization: "{{ authorizationHeader }}",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.blob();
    return result;
}

query({ inputs: {{ providerInputs.asObj.inputs }} }).then((response) => {
    // Returns a byte object of the Audio wavform. Use it directly!
});
{% else %}
async function query(data) {
	const response = await fetch(
		"{{ fullUrl }}",
		{
			headers: {
				Authorization: "{{ authorizationHeader }}",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
    const result = await response.json();
    return result;
}

query({ inputs: {{ providerInputs.asObj.inputs }} }).then((response) => {
    console.log(JSON.stringify(response));
});
{% endif %} `,textToImage:`async function query(data) {
	const response = await fetch(
		"{{ fullUrl }}",
		{
			headers: {
				Authorization: "{{ authorizationHeader }}",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.blob();
	return result;
}

query({ inputs: {{ providerInputs.asObj.inputs }} }).then((response) => {
    // Use image
});`,zeroShotClassification:`async function query(data) {
    const response = await fetch(
		"{{ fullUrl }}",
        {
            headers: {
				Authorization: "{{ authorizationHeader }}",
                "Content-Type": "application/json",
            },
            method: "POST",
            body: JSON.stringify(data),
        }
    );
    const result = await response.json();
    return result;
}

query({
    inputs: {{ providerInputs.asObj.inputs }},
    parameters: { candidate_labels: ["refund", "legal", "faq"] }
}).then((response) => {
    console.log(JSON.stringify(response));
});`},"huggingface.js":{basic:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

const output = await client.{{ methodName }}({
	model: "{{ model.id }}",
	inputs: {{ inputs.asObj.inputs }},
	provider: "{{ provider }}",
});

console.log(output);`,basicAudio:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

const data = fs.readFileSync({{inputs.asObj.inputs}});

const output = await client.{{ methodName }}({
	data,
	model: "{{ model.id }}",
	provider: "{{ provider }}",
});

console.log(output);`,basicImage:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

const data = fs.readFileSync({{inputs.asObj.inputs}});

const output = await client.{{ methodName }}({
	data,
	model: "{{ model.id }}",
	provider: "{{ provider }}",
});

console.log(output);`,conversational:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

const chatCompletion = await client.chatCompletion({
    provider: "{{ provider }}",
    model: "{{ model.id }}",
{{ inputs.asTsString }}
});

console.log(chatCompletion.choices[0].message);`,conversationalStream:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

let out = "";

const stream = await client.chatCompletionStream({
    provider: "{{ provider }}",
    model: "{{ model.id }}",
{{ inputs.asTsString }}
});

for await (const chunk of stream) {
	if (chunk.choices && chunk.choices.length > 0) {
		const newContent = chunk.choices[0].delta.content;
		out += newContent;
		console.log(newContent);
	}  
}`,textToImage:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

const image = await client.textToImage({
    provider: "{{ provider }}",
    model: "{{ model.id }}",
	inputs: {{ inputs.asObj.inputs }},
	parameters: { num_inference_steps: 5 },
});
/// Use the generated image (it's a Blob)`,textToVideo:`import { InferenceClient } from "@huggingface/inference";

const client = new InferenceClient("{{ accessToken }}");

const image = await client.textToVideo({
    provider: "{{ provider }}",
    model: "{{ model.id }}",
	inputs: {{ inputs.asObj.inputs }},
});
// Use the generated video (it's a Blob)`},openai:{conversational:`import { OpenAI } from "openai";

const client = new OpenAI({
	baseURL: "{{ baseUrl }}",
	apiKey: "{{ accessToken }}",
});

const chatCompletion = await client.chat.completions.create({
	model: "{{ providerModelId }}",
{{ inputs.asTsString }}
});

console.log(chatCompletion.choices[0].message);`,conversationalStream:`import { OpenAI } from "openai";

const client = new OpenAI({
	baseURL: "{{ baseUrl }}",
	apiKey: "{{ accessToken }}",
});

let out = "";

const stream = await client.chat.completions.create({
    provider: "{{ provider }}",
    model: "{{ model.id }}",
{{ inputs.asTsString }}
});

for await (const chunk of stream) {
	if (chunk.choices && chunk.choices.length > 0) {
		const newContent = chunk.choices[0].delta.content;
		out += newContent;
		console.log(newContent);
	}  
}`}},python:{fal_client:{textToImage:`{% if provider == "fal-ai" %}
import fal_client

result = fal_client.subscribe(
    "{{ providerModelId }}",
    arguments={
        "prompt": {{ inputs.asObj.inputs }},
    },
)
print(result)
{% endif %} `},huggingface_hub:{basic:`result = client.{{ methodName }}(
    inputs={{ inputs.asObj.inputs }},
    model="{{ model.id }}",
)`,basicAudio:'output = client.{{ methodName }}({{ inputs.asObj.inputs }}, model="{{ model.id }}")',basicImage:'output = client.{{ methodName }}({{ inputs.asObj.inputs }}, model="{{ model.id }}")',conversational:`completion = client.chat.completions.create(
    model="{{ model.id }}",
{{ inputs.asPythonString }}
)

print(completion.choices[0].message) `,conversationalStream:`stream = client.chat.completions.create(
    model="{{ model.id }}",
{{ inputs.asPythonString }}
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="") `,documentQuestionAnswering:`output = client.document_question_answering(
    "{{ inputs.asObj.image }}",
    question="{{ inputs.asObj.question }}",
    model="{{ model.id }}",
) `,imageToImage:`# output is a PIL.Image object
image = client.image_to_image(
    "{{ inputs.asObj.inputs }}",
    prompt="{{ inputs.asObj.parameters.prompt }}",
    model="{{ model.id }}",
) `,importInferenceClient:`from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="{{ provider }}",
    api_key="{{ accessToken }}",
)`,textToImage:`# output is a PIL.Image object
image = client.text_to_image(
    {{ inputs.asObj.inputs }},
    model="{{ model.id }}",
) `,textToVideo:`video = client.text_to_video(
    {{ inputs.asObj.inputs }},
    model="{{ model.id }}",
) `},openai:{conversational:`from openai import OpenAI

client = OpenAI(
    base_url="{{ baseUrl }}",
    api_key="{{ accessToken }}"
)

completion = client.chat.completions.create(
    model="{{ providerModelId }}",
{{ inputs.asPythonString }}
)

print(completion.choices[0].message) `,conversationalStream:`from openai import OpenAI

client = OpenAI(
    base_url="{{ baseUrl }}",
    api_key="{{ accessToken }}"
)

stream = client.chat.completions.create(
    model="{{ providerModelId }}",
{{ inputs.asPythonString }}
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")`},requests:{basic:`def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": {{ providerInputs.asObj.inputs }},
}) `,basicAudio:`def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers={"Content-Type": "audio/flac", **headers}, data=data)
    return response.json()

output = query({{ providerInputs.asObj.inputs }})`,basicImage:`def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers={"Content-Type": "image/jpeg", **headers}, data=data)
    return response.json()

output = query({{ providerInputs.asObj.inputs }})`,conversational:`def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
{{ providerInputs.asJsonString }}
})

print(response["choices"][0]["message"])`,conversationalStream:`def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload, stream=True)
    for line in response.iter_lines():
        if not line.startswith(b"data:"):
            continue
        if line.strip() == b"data: [DONE]":
            return
        yield json.loads(line.decode("utf-8").lstrip("data:").rstrip("/n"))

chunks = query({
{{ providerInputs.asJsonString }},
    "stream": True,
})

for chunk in chunks:
    print(chunk["choices"][0]["delta"]["content"], end="")`,documentQuestionAnswering:`def query(payload):
    with open(payload["image"], "rb") as f:
        img = f.read()
        payload["image"] = base64.b64encode(img).decode("utf-8")
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": {
        "image": "{{ inputs.asObj.image }}",
        "question": "{{ inputs.asObj.question }}",
    },
}) `,imageToImage:`def query(payload):
    with open(payload["inputs"], "rb") as f:
        img = f.read()
        payload["inputs"] = base64.b64encode(img).decode("utf-8")
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

image_bytes = query({
{{ providerInputs.asJsonString }}
})

# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes)) `,importRequests:`{% if importBase64 %}
import base64
{% endif %}
{% if importJson %}
import json
{% endif %}
import requests

API_URL = "{{ fullUrl }}"
headers = {"Authorization": "{{ authorizationHeader }}"}`,tabular:`def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

response = query({
    "inputs": {
        "data": {{ providerInputs.asObj.inputs }}
    },
}) `,textToAudio:`{% if model.library_name == "transformers" %}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

audio_bytes = query({
    "inputs": {{ providerInputs.asObj.inputs }},
})
# You can access the audio with IPython.display for example
from IPython.display import Audio
Audio(audio_bytes)
{% else %}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

audio, sampling_rate = query({
    "inputs": {{ providerInputs.asObj.inputs }},
})
# You can access the audio with IPython.display for example
from IPython.display import Audio
Audio(audio, rate=sampling_rate)
{% endif %} `,textToImage:`{% if provider == "hf-inference" %}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

image_bytes = query({
    "inputs": {{ providerInputs.asObj.inputs }},
})

# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))
{% endif %}`,zeroShotClassification:`def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": {{ providerInputs.asObj.inputs }},
    "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
}) `,zeroShotImageClassification:`def query(data):
    with open(data["image_path"], "rb") as f:
        img = f.read()
    payload={
        "parameters": data["parameters"],
        "inputs": base64.b64encode(img).decode("utf-8")
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "image_path": {{ providerInputs.asObj.inputs }},
    "parameters": {"candidate_labels": ["cat", "dog", "llama"]},
}) `}},sh:{curl:{basic:`curl {{ fullUrl }} \\
    -X POST \\
    -H 'Authorization: {{ authorizationHeader }}' \\
    -H 'Content-Type: application/json' \\
    -d '{
{{ providerInputs.asCurlString }}
    }'`,basicAudio:`curl {{ fullUrl }} \\
    -X POST \\
    -H 'Authorization: {{ authorizationHeader }}' \\
    -H 'Content-Type: audio/flac' \\
    --data-binary @{{ providerInputs.asObj.inputs }}`,basicImage:`curl {{ fullUrl }} \\
    -X POST \\
    -H 'Authorization: {{ authorizationHeader }}' \\
    -H 'Content-Type: image/jpeg' \\
    --data-binary @{{ providerInputs.asObj.inputs }}`,conversational:`curl {{ fullUrl }} \\
    -H 'Authorization: {{ authorizationHeader }}' \\
    -H 'Content-Type: application/json' \\
    -d '{
{{ providerInputs.asCurlString }},
        "stream": false
    }'`,conversationalStream:`curl {{ fullUrl }} \\
    -H 'Authorization: {{ authorizationHeader }}' \\
    -H 'Content-Type: application/json' \\
    -d '{
{{ providerInputs.asCurlString }},
        "stream": true
    }'`,zeroShotClassification:`curl {{ fullUrl }} \\
    -X POST \\
    -d '{"inputs": {{ providerInputs.asObj.inputs }}, "parameters": {"candidate_labels": ["refund", "legal", "faq"]}}' \\
    -H 'Content-Type: application/json' \\
    -H 'Authorization: {{ authorizationHeader }}'`}}},bs=["huggingface_hub","fal_client","requests","openai"],ys=["fetch","huggingface.js","openai"],ws=["curl"],vs={js:[...ys],python:[...bs],sh:[...ws]},_s=(e,i,t)=>{var a,n;return((n=(a=it[e])==null?void 0:a[i])==null?void 0:n[t])!==void 0},_e=(e,i,t)=>{var n,o;const a=(o=(n=it[e])==null?void 0:n[i])==null?void 0:o[t];if(!a)throw new Error(`Template not found: ${e}/${i}/${t}`);return s=>new Kn(a).render({...s})},ks=_e("python","huggingface_hub","importInferenceClient"),xs=_e("python","requests","importRequests"),at={"audio-classification":"audio_classification","audio-to-audio":"audio_to_audio","automatic-speech-recognition":"automatic_speech_recognition","document-question-answering":"document_question_answering","feature-extraction":"feature_extraction","fill-mask":"fill_mask","image-classification":"image_classification","image-segmentation":"image_segmentation","image-to-image":"image_to_image","image-to-text":"image_to_text","object-detection":"object_detection","question-answering":"question_answering","sentence-similarity":"sentence_similarity",summarization:"summarization","table-question-answering":"table_question_answering","tabular-classification":"tabular_classification","tabular-regression":"tabular_regression","text-classification":"text_classification","text-generation":"text_generation","text-to-image":"text_to_image","text-to-speech":"text_to_speech","text-to-video":"text_to_video","token-classification":"token_classification",translation:"translation","visual-question-answering":"visual_question_answering","zero-shot-classification":"zero_shot_classification","zero-shot-image-classification":"zero_shot_image_classification"},nt={"automatic-speech-recognition":"automaticSpeechRecognition","feature-extraction":"featureExtraction","fill-mask":"fillMask","image-classification":"imageClassification","question-answering":"questionAnswering","sentence-similarity":"sentenceSimilarity",summarization:"summarization","table-question-answering":"tableQuestionAnswering","text-classification":"textClassification","text-generation":"textGeneration","text2text-generation":"textGeneration","token-classification":"tokenClassification",translation:"translation"},S=(e,i)=>(t,a,n,o,s)=>{var y;t.pipeline_tag&&["text-generation","image-text-to-text"].includes(t.pipeline_tag)&&t.tags.includes("conversational")&&(e=s!=null&&s.streaming?"conversationalStream":"conversational",i=Is);const d=i?i(t,s):{inputs:te(t)},u=Xe(o??t.id,{accessToken:a,provider:n,...d},{chatCompletion:e.includes("conversational"),task:t.pipeline_tag});let c=d;const f=u.info.body;if(typeof f=="string")try{c=JSON.parse(f)}catch(_){console.error("Failed to parse body as JSON",_)}const p={accessToken:a,authorizationHeader:(y=u.info.headers)==null?void 0:y.Authorization,baseUrl:Ts(u.url,"/chat/completions"),fullUrl:u.url,inputs:{asObj:d,asCurlString:B(d,"curl"),asJsonString:B(d,"json"),asPythonString:B(d,"python"),asTsString:B(d,"ts")},providerInputs:{asObj:c,asCurlString:B(c,"curl"),asJsonString:B(c,"json"),asPythonString:B(c,"python"),asTsString:B(c,"ts")},model:t,provider:n,providerModelId:o??t.id};return kn.map(_=>vs[_].map(k=>{if(!_s(_,k,e))return;const C=_e(_,k,e);if(k==="huggingface_hub"&&e.includes("basic")){if(!(t.pipeline_tag&&t.pipeline_tag in at))return;p.methodName=at[t.pipeline_tag]}if(k==="huggingface.js"&&e.includes("basic")){if(!(t.pipeline_tag&&t.pipeline_tag in nt))return;p.methodName=nt[t.pipeline_tag]}let T=C(p).trim();if(T)return k==="huggingface_hub"?T=`${ks({...p})}

${T}`:k==="requests"&&(T=`${xs({...p,importBase64:T.includes("base64"),importJson:T.includes("json.")})}

${T}`),{language:_,client:k,content:T}}).filter(k=>k!==void 0)).flat()},As=e=>JSON.parse(te(e)),Ss=e=>{const i=JSON.parse(te(e));return{inputs:i.image,parameters:{prompt:i.prompt}}},Is=(e,i)=>({messages:(i==null?void 0:i.messages)??te(e),...i!=null&&i.temperature?{temperature:i==null?void 0:i.temperature}:void 0,max_tokens:(i==null?void 0:i.max_tokens)??500,...i!=null&&i.top_p?{top_p:i==null?void 0:i.top_p}:void 0}),me={"audio-classification":S("basicAudio"),"audio-to-audio":S("basicAudio"),"automatic-speech-recognition":S("basicAudio"),"document-question-answering":S("documentQuestionAnswering",As),"feature-extraction":S("basic"),"fill-mask":S("basic"),"image-classification":S("basicImage"),"image-segmentation":S("basicImage"),"image-text-to-text":S("conversational"),"image-to-image":S("imageToImage",Ss),"image-to-text":S("basicImage"),"object-detection":S("basicImage"),"question-answering":S("basic"),"sentence-similarity":S("basic"),summarization:S("basic"),"tabular-classification":S("tabular"),"tabular-regression":S("tabular"),"table-question-answering":S("basic"),"text-classification":S("basic"),"text-generation":S("basic"),"text-to-audio":S("textToAudio"),"text-to-image":S("textToImage"),"text-to-speech":S("textToAudio"),"text-to-video":S("textToVideo"),"text2text-generation":S("basic"),"token-classification":S("basic"),translation:S("basic"),"zero-shot-classification":S("zeroShotClassification"),"zero-shot-image-classification":S("zeroShotImageClassification")};function Es(e,i,t,a,n){var o;return e.pipeline_tag&&e.pipeline_tag in me?((o=me[e.pipeline_tag])==null?void 0:o.call(me,e,i,t,a,n))??[]:[]}function B(e,i){switch(i){case"curl":return ot(B(e,"json"));case"json":return JSON.stringify(e,null,4).split(`
`).slice(1,-1).join(`
`);case"python":return ot(Object.entries(e).map(([t,a])=>{const n=JSON.stringify(a,null,4).replace(/"/g,'"');return`${t}=${n},`}).join(`
`));case"ts":return ke(e).split(`
`).slice(1,-1).join(`
`);default:throw new Error(`Unsupported format: ${i}`)}}function ke(e,i){return i=i??0,typeof e!="object"||e===null?JSON.stringify(e):Array.isArray(e)?`[
${e.map(o=>{const s=ke(o,i+1);return`${" ".repeat(4*(i+1))}${s},`}).join(`
`)}
${" ".repeat(4*i)}]`:`{
${Object.entries(e).map(([n,o])=>{const s=ke(o,i+1),d=/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(n)?n:`"${n}"`;return`${" ".repeat(4*(i+1))}${d}: ${s},`}).join(`
`)}
${" ".repeat(4*i)}}`}function ot(e){return e.split(`
`).map(i=>" ".repeat(4)+i).join(`
`)}function Ts(e,i){return e.endsWith(i)?e.slice(0,-i.length):e}function Z(e,i){if(!e)throw new Error(i)}const Cs=()=>({type:"WORKER_READY",output:{initialized:!0}}),Us=(e,i)=>({status:"complete",output:{classification:e,description:i},type:"CATEGORIZE_IMAGE"}),rt=e=>({status:"error",output:e,type:"CATEGORIZE_IMAGE"}),Os=e=>({status:"complete",output:{isValid:e},type:"TOKEN_VALIDATION"}),Ms=e=>({status:"error",output:e,type:"TOKEN_VALIDATION"}),P=console.log.bind(console,"[ImageRecognizerWorker]"),xe=console.error.bind(console,"[ImageRecognizerWorker]"),Q=e=>{P("Sending message to main thread:",e),self.postMessage(e)};let F=null;const st=async e=>{P("Initializing client with token:",e?"***":"empty");try{return Z(e,"Token is required"),Z(e.startsWith("hf_"),"Invalid token format"),F=new fs(e),Z(F,"Client initialization failed"),await F.textGeneration({model:"gpt2",inputs:"Hello",parameters:{max_new_tokens:1,do_sample:!1}}),P("Client initialized and validated successfully"),!0}catch(i){return F=null,P("Client initialization error:",i),!1}};self.addEventListener("message",async e=>{P("Received message from main thread:",e.data);try{switch(e.data.type){case"VALIDATE_TOKEN":{P("Validating token...");const{token:i}=e.data;try{const t=await st(i);P("Token validation result:",t),Q(Os(t))}catch(t){const a=t instanceof Error?t.message:String(t);xe("Token validation error:",t),Q(Ms(a))}break}case"CATEGORIZE_IMAGE":{P("Categorizing image...");const{imageData:i}=e.data,t=e.data.token||"";if(!F&&!await st(t)){Q(rt("Failed to initialize client: Invalid token"));return}try{Z(i,"Image data is required"),Z(i instanceof Blob,"Image data must be a Blob"),Z(F,"Client is not initialized");const[a,n]=await Promise.all([F.imageToText({data:i,model:"Salesforce/blip-image-captioning-base"}),F.zeroShotImageClassification({inputs:i,model:"openai/clip-vit-large-patch14",parameters:{candidate_labels:["fruits-vegetables","dairy-and-eggs","meat-fish","grains","canned-goods","spices","snacks","beverages","other"]},provider:"hf-inference"})]);P("Classification result:",n),P("Image to text result:",a),Q(Us(n,a))}catch(a){const n=a instanceof Error?a.message:String(a);xe("Classification error:",a),Q(rt(n))}break}default:throw new Error(`Unknown message type: ${e.data.type}`)}}catch(i){const t=i instanceof Error?i.message:String(i);xe("Handler error:",i),Q({status:"error",output:t,type:e.data.type})}}),Q(Cs())})();
