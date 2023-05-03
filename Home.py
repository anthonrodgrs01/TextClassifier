# import streamlit as st
#
# # Set the page configuration
# st.set_page_config(
#     page_title="Text Classification App",
#     page_icon="📚"
# )
#
# st.balloons()
#
# def app():
#
#     # Set the page title
#     st.title('Welcome to the Text Classification App!')
#
#     # Add a short description
#     st.markdown('Text classification is the process of categorizing text data into different classes or categories based on its content. It has many applications such as sentiment analysis, spam filtering, and topic modeling.')
#
#     # Create two columns for the app features and how to use sections
#     col1, col2 = st.columns(2)
#
#     # Add a section for the app features
#     with col1:
#         st.markdown('<h2>Features</h2>', unsafe_allow_html=True)
#         st.markdown('<ul><li>Free and open source.</li><li>Text classification using CNN</li><li>File upload support for text, PDF, Word files</li><li>YouTube Comment Sentiment Analysis</li></ul>', unsafe_allow_html=True)
#
#     # Add a section for how to use the app
#     with col2:
#         st.markdown('<h2>How to Use?</h2>', unsafe_allow_html=True)
#         st.markdown('<ol><li>Enter text in the input box and Press "Enter" to classify sentiment.</li><li>Or drag and drop a text, PDF, or Word file to classify sentiment.</li></ol>', unsafe_allow_html=True)
#
#     # Add a section for the app information
#     st.markdown('<h2>About the App</h2>', unsafe_allow_html=True)
#     st.markdown('<p>This app is built using Streamlit, a free and open-source Python library for building data apps. Streamlit makes it easy to create beautiful and interactive apps with minimal code.</p>', unsafe_allow_html=True)
#
#
#
# if __name__ == "__main__":
#     app()


import streamlit as st
#from streamlit_lottie import st_lottie
import requests

# Set the page configuration
st.set_page_config(
    page_title="Text Classification App",
    page_icon="📚"
)

st.balloons()

def app():

    # Set the page title
    st.title('Welcome to the Text Classification App!')

    # Add a short description
    st.markdown('Text classification is the process of categorizing text data into different classes or categories based on its content. It has many applications such as sentiment analysis, spam filtering, and topic modeling.')

    # Create two columns for the app features and how to use sections
    col1, col2 = st.columns(2)

    # Add a section for the app features
    with col1:
        st.markdown('<h2>Features</h2>', unsafe_allow_html=True)
        st.markdown('<ul><li>Free and open source.</li><li>Text classification using CNN</li><li>File upload support for text, PDF, Word files</li><li>YouTube Comment Sentiment Analysis</li></ul>', unsafe_allow_html=True)



    # Add a section for how to use the app
    with col2:
        st.markdown('<h2>How to Use?</h2>', unsafe_allow_html=True)
        st.markdown('<ol><li>Enter text in the input box and Press "Enter" to classify sentiment.</li><li>Or drag and drop a text, PDF, or Word file to classify sentiment.</li></ol>', unsafe_allow_html=True)

    # Load and display the Lottie animation
    # st_lottie(load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_x5Tg8Cv6dM.json"), speed=1, width=300,height=300, key="hello")



    col1, col2, col3 = st.columns(3)


    with col2:
        st_lottie(load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_x5Tg8Cv6dM.json"), speed=1, width=235,height=235, key="hello")


    # Add a section for the app information
    st.markdown('<h2>About the App</h2>', unsafe_allow_html=True)
    st.markdown('<p>This app is built using Streamlit, a free and open-source Python library for building data apps. Streamlit makes it easy to create beautiful and interactive apps with minimal code.</p>', unsafe_allow_html=True)



# Function to load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if __name__ == "__main__":
    app()




