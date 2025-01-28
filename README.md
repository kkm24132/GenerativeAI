## GenerativeAI

For Primers - I suggest to focus on solid foundation before just jumping on to advanced areas of Data Science, always. It helps in building our understanding better and more importantly "applicability"!!

Here are few key pointers for initial foundation to note:
- Master the Fundamentals: Machine Learning (ML) Basics: Understand concepts like supervised learning, unsupervised learning, overfitting, regularization, and optimization techniques. Learn to implement ML algorithms, such as decision trees, SVMs, random forest and neural networks etc.
- Core Software Engineering Skills: AI engineering is as much about writing scalable, maintainable code as it is about developing models. Be proficient in Python, and also understand the importance of unit testing, debugging, and deploying applications on cloud (Docker, Kubernetes).
- Delve into Embedding Models: Embedding models (e.g., Word2Vec, BERT) are key to NLP applications. Learn how embeddings represent words or data in a dense vector space and how they capture semantic meaning. Experiment with building and fine-tuning models, understanding loss functions, and embedding techniques for both NLP and non-NLP use cases.
- Understand GenAI beyond Chatbots: GenAI is not just about creating RAG chatbots. It includes models like GPT, BERT, and Stable Diffusion, used for generating text, images, and more. You need to grasp the Deep Learning architecture behind these models (Transformers, self attention, positional encoding), training processes, and applications, as well as their limitations (e.g., hallucinations, bias).
- Get Hands-On Experience: Apply your knowledge to specific use cases. Build projects that include end-to-end pipelines (data collection, preprocessing, model fine-tuning, GenAI model evaluation, and deployment).
- Learn about GenAI’s Ethical Implications: Generative AI has its pros (automation, personalization) but also has serious concerns (hallucinations, deepfakes, biases). Understand these challenges and how to mitigate them.

### Recommended Learning / Top Courses to Learn about Generative AI

- **Stanford University: Deep Learning for Generative Models**
  - Covers the basics of Deep Learning and how to apply it to Generative models.
  - At Stanford University, an informative course is offered on the fundamental principles of deep learning and its application to generative models. The curriculum delves into various types of deep learning architectures, including CNNs and RNNs, and teaches the methodology of using them for creating generative models. Please [refer here](https://deepgenerativemodels.github.io/2019/)
  - Key topics covered:
    - Introduction to Deep Learning
    - Convolutional Neural Networks (CNNs)
    - Recurrent Neural Networks (RNNs)
    - Generative Deep Learning Models
    - Applications of Generative Deep Learning 
- **MIT OpenCourseWare**
  - Covers the basics of Generative models for text and images
  - The MIT OpenCourseWare offers a comprehensive course on generative models for text and images. The course covers the fundamentals of various generative models, including RNNs and CNNs, and provides instruction on how to utilize them to produce authentic text and images. Please [refer here](https://ocw.mit.edu/courses/16-412j-cognitive-robotics-spring-2016/resources/advanced-lecture-3-image-classification-via-deep-learning/)
  - Key topics covered:
    - Introduction to Generative Models
    - Recurrent Neural Networks (RNNs)
    - Convolutional Neural Networks (CNNs)
    - Generating Text with RNNs
    - Generating Images with CNNs  
- **Google: Intro Generative AI courses**
  - Please [refer here](https://www.cloudskillsboost.google/paths/118)   
- **Books on Generative AI and NLP**
  - [Natural Language Processing with Transformers: Building Language Applications with Hugging Face](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)
  - [Generative AI with Python and TensorFlow 2: Create images, text, and music with VAEs, GANs, LSTMs, Transformer models](https://www.amazon.com.au/Generative-AI-Python-TensorFlow-Transformer/dp/1800200889)
  - [Transformers for Natural Language Processing - Second Edition: Build, train, and fine-tune deep neural network architectures for NLP with Python, Hugging Face, and OpenAI's GPT-3, ChatGPT, and GPT-4](https://www.amazon.com/Transformers-Natural-Language-Processing-architectures/dp/1803247339)
 
- **Udemy**
  - [Generative AI for Creative Applications](https://www.udemy.com/course/generative-ai/)
  - Key topics covered:
    - Introduction to Generative AI for Creative Applications
    - Generative Adversarial Networks (GANs)
    - Variational Autoencoders (VAEs)
    - Creating Art with GANs
    - Creating Music with VAEs  
- **Udacity**
  - Please [refer here](https://www.udacity.com/course/building-generative-adversarial-networks--cd1823)
  - Key topics covered:
    - Introduction to GANs
    - The Generator
    - The Discriminator
    - Training GANs
    - Applications of GANs  

- **Courses from NVIDIA**
  - Generative AI Explained (Learn how to:) [refer here](https://lnkd.in/gBb3peXi)
    - Define Generative AI and explain how Generative AI works
    - Describe various Generative AI applications
    - Explain the challenges and opportunities in Generative AI
  - AI for All: From Basics to GenAI Practice [refer here](https://lnkd.in/gXmmnC4G)
    - AI impacts industries like healthcare and autonomous vehicles
    - From machine learning to generative AI
    - GenAI creates music, images, and videos
  - Getting Started with AI on Jetson Nano (Learn how to:) [refer here](https://lnkd.in/gnmrhBJm)
    - Set up your Jetson Nano and camera
    - Collect image data for classification models
    - Annotate image data for regression models
    - Train a neural network on your data to create your models
  - Building A Brain in 10 Minutes [refer here](https://lnkd.in/gCaA-XKp)
    - How neural networks use data to learn
    - Understand the math behind a neuron
  - Building Video AI Applications on Jetson Nano: [refer here](https://lnkd.in/gNffgw5C)
    - Create DeepStream pipelines for video processing
    - Handle multiple video streams
    - Use alternate inference engines like YOLO
  - Building RAG Agents with LLMs [refer here](https://lnkd.in/gcK2ZJ4a)
    - Explore scalable deployment strategies
    - Learn about microservices and development
    - Experiment with LangChain paradigms for dialog management
    - Practice with state-of-the-art models
  - Accelerate Data Science Workflows with Zero Code Changes [refer here](https://lnkd.in/gF7eVk2V)
    - Learn the benefits of unified CPU and GPU workflows
    - GPU-accelerate data processing and machine learning
    - See faster processing times with GPU
  - Introduction to AI in the Data Center [refer here](https://lnkd.in/gKTS6uMS)
    - Basics of AI, machine learning, and GPU architecture
    - Deep learning frameworks and AI workload deployment
    - Learn about multi-system AI clusters and infrastructure planning

### Attention mechanism

The "Attention Mechanism" in transformers is the backbone of all Generative AI models. It may seem magical to someone jumping into the subject, but this concept did not appear from nothing. To understand the evolution of Attention mechanism, here are few foundational papers that explain the concepts that paved the way for modern transformer architectures.
- [Long Short-term memory RNN](https://arxiv.org/abs/2105.06756)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Sequence to Sequence Learning with Neural Nets](https://arxiv.org/abs/1409.3215)


### References

- [TeachFX Explainer Video](https://www.youtube.com/watch?v=QIL6mQGDTME) - Personalized feedback for teachers. Reflect on rich, research-based, classroom-level insights. Support every teacher to superpower their own practice and transform how learning happens.
- Text Based Applications:
  - ChatGPT (OpenAI): Versatile conversational AI for general assistance, creative writing, and problem-solving.
  - Jasper: AI-powered content creation tool for marketing, blogs, and social media.
  - Writesonic: AI writer for generating articles, product descriptions, and advertisements.
  - Copy.ai: Tool for copywriting, email drafts, and brainstorming ideas.
  - Claude (Anthropic): Ethical, thoughtful conversational AI alternative to GPT models.
- Developer & Automation Tools:
  - OpenAI API (GPT models): Integration of language models into applications.
  - LangChain: Framework for building applications that leverage LLMs in workflows (e.g., multi-step reasoning).
  - Pinecone / Weaviate: Vector databases for semantic search and context management.
  - LlamaIndex (formerly GPT Index): Tools for connecting LLMs with data sources and custom document contexts.
  - DeepSeek API and coder: AI-powered code generation and analysis [Paper:DeepSeek-Coder](https://arxiv.org/abs/2401.14196) and [Git Ref](https://github.com/deepseek-ai/DeepSeek-Coder)


