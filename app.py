import streamlit as st
from utils.pdf_utils import extract_text_from_pdf
from model_utils.sentiment import load_sentiment_resources, predict_sentiment
from model_utils.summarization import load_summarization_resources, generate_summary

def main():
    st.title('AI Powered Research Assistant')
    uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'])
    input_text = ''

    if uploaded_file is not None:
        extracted = extract_text_from_pdf(uploaded_file)
        st.text_area('Extracted Text from PDF', extracted, height=150, disabled=True)
        input_text = extracted
    else:
        input_text = st.text_area('Or paste your text here', '', height=200)

    feature = st.selectbox('Select a feature to analyze:', ['Summarization', 'Q&A', 'NER', 'Sentiment Analysis'])

    question = ''
    if feature == 'Q&A':
        question = st.text_input('Enter your question related to the context')

    if st.button('Analyze'):
        if not input_text.strip():
            st.warning('Please provide some input text or upload a PDF.')
            return

        if feature == 'Sentiment Analysis':
            vocab_sent, model_sent = load_sentiment_resources()
            sentiment, conf = predict_sentiment(input_text, vocab_sent, model_sent)
            st.success(f'Sentiment: {sentiment} (Confidence: {conf:.2f})')

        elif feature == 'Summarization':
            vocab_sum, inv_vocab_sum, model_sum = load_summarization_resources()
            summary = generate_summary(input_text, vocab_sum, inv_vocab_sum, model_sum)
            st.subheader('Summary:')
            st.write(summary)

        elif feature == 'Q&A':
            st.info('Q&A model integration not implemented yet.')

        elif feature == 'NER':
            st.info('NER model integration not implemented yet.')

if __name__ == '__main__':
    main()
