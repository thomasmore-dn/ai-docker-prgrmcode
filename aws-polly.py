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
