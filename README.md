# Assignment 2 – AWS

## Part 1: Managed AI services in AWS

### Introduction to Amazon Polly

Amazon Polly is a Text-to-Speech (TTS) service provided by Amazon Web Services (AWS). It converts written text into lifelike speech, allowing developers to create applications that can speak and engage users through natural-sounding voices.

### 1. Pricing

Amazon Polly's pricing is based on the number of characters converted from text to speech. Pricing details can be found on the [Amazon Polly Pricing Page](https://aws.amazon.com/polly/pricing/). It offers a free tier for the first 12 months, making it accessible for experimentation and small-scale projects.

#### Amazon Polly Pricing Details

With Amazon Polly, you only pay for what you use. You are charged based on the number of characters of text that you convert either to speech or to Speech Marks metadata. In addition, you can cache and replay Amazon Polly’s generated speech at no additional cost. It's easy to get started with the Amazon Polly Free Tier; try it today.

**Pricing:**

- **PAY-AS-YOU-GO MODEL:** You are billed monthly for the number of characters of text that you processed. Amazon Polly’s Standard voices are priced at $4.00 per 1 million characters for speech or Speech Marks requests (when outside the free tier). Amazon Polly’s Neural voices are priced at $16.00 per 1 million characters for speech or Speech Marks requested (when outside the free tier).
- **Free Tier:** For Amazon Polly’s Standard voices, the free tier includes 5 million characters per month for speech or Speech Marks requests, for the first 12 months, starting from your first request for speech. For Neural voices, the free tier includes 1 million characters per month for speech or Speech Marks requests, for the first 12 months, starting from your first request for speech.

### 2. Console Usage

Amazon Polly can be used directly from the AWS Management Console.

**To synthesize speech using plain text input (console):**

1. Navigate to the Polly Service: Go to the AWS Management Console, and in the services search bar, type "Polly" and select it.
2. Create a New Speech Synthesis Task:
   - After logging on to the Amazon Polly console, choose Try Amazon Polly, and then choose the Text-to-Speech tab.
   - Turn off SSML.
   - Type or paste the text you want to convert to speech into the input box.
   - Under Engine, choose Standard, Neural, or Long Form.
   - Choose a language and AWS Region, then choose a voice. If you choose Neural for Engine, only the languages and voices that support NTTS are available. All Standard and Long Form voices are disabled.
3. Listen to the Generated Speech:
   - Once the synthesis task is complete, you can listen to the generated speech directly from the console.
   - To listen to the speech immediately, choose Listen.
4. To save the speech to a file, do one of the following:
   - Choose Download.
   - To change to a different file format, expand Additional settings, turn on Speech file format settings, choose the file format that you want, and then choose Download.
5. To save the speech to an S3 bucket, do one of the following:
   - Click on the Save to S3 button.
   - Enter the S3 output bucket name.
   - Click the Save to S3 button.

### 3. Build a simple python application in which you use the service

**Step 1: Install Boto3 Library**

```bash
pip install boto3
```

**Step 2: Set Up AWS Credentials**
Ensure that AWS credentials are configured either through environment variables or AWS CLI.

**Step 3: Python Script (aws-polly.py):**

```python
import boto3

# Initialize Polly client
polly_client = boto3.client('polly')

# Specify text to be converted
text_to_convert = "Hello, welcome to the world of Amazon Polly. This is a sample text."

# Specify voice ID and language
voice_id = 'Joanna'
language_code = 'en-US'

# Request Polly to synthesize speech
response = polly_client.synthesize_speech(
    Text=text_to_convert,
    VoiceId=voice_id,
    OutputFormat='mp3',
    LanguageCode=language_code
)

# Save the generated speech to an MP3 file
with open('output.mp3', 'wb') as file:
    file.write(response['AudioStream'].read())

print("Speech synthesis completed. Check the 'output.mp3' file.")
```

- The script uses the Boto3 library to interact with Polly.
- It specifies the text, voice, and language for synthesis.
- The generated speech is saved as an MP3 file (output.mp3).

![](aws-polly.png)

Output file created from AWS Polly with python script: [output.mp3](output.mp3)

# Part 2: Amazon Sagemaker

Use the guided tutorial ["Train an ML model"](https://aws.amazon.com/tutorials/machine-learning-tutorial-train-a-model/) to get familiar with AWS Sagemaker.

When you finished the tutorial, create a solution for an ML model of your choice.

## ML Model created with Amazon AWS SageMaker Studio

- [BinomialLogisticRegression_AWS-Sagemaker-Studio.ipynb](BinomialLogisticRegression_AWS-Sagemaker-Studio.ipynb)
