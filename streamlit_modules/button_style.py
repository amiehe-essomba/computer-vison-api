def button_style(st, name: str = "run", class_ : str = "custom-button"):
    #import streamlit as st

    # Define a custom CSS class for our button
    button_style = """
        .custom-button {
            background-color: #4CAF50; /* Green background color */
            border: none; /* Remove border */
            color: white; /* White text color */
            padding: 10px 20px; /* Padding */
            text-align: center; /* Center text */
            display: inline-block; /* Display as inline block */
            font-size: 16px; /* Font size */
            border-radius: 50px; /* Make it a circle shape */
        }
        """

    """
    # Add the custom CSS to the Streamlit app
    st.markdown(button_style, unsafe_allow_html=True)

    # Create a custom button using HTML
    custom_button = f'<button class={class_}>{name}</button>'
    #st.markdown(custom_button, unsafe_allow_html=True)
    """

    st.write(
        f'<style>div.stButton > button{{background-color: white; color: darkgreen; \
            padding: 10px 20px, text-align: center;\
            display: inline-block;\
            font-size: 16px;\
            border-radius: 50px;\
            font-weight: bolder}} </style>',
        unsafe_allow_html=True
    )

    col, _, _, _, _ = st.columns(5)

    with col:
        run = st.button(name)

    return run