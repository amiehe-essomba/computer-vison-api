def conclusion(st):
    import streamlit as st

    st.write('<style>{}</style>'.format(styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Conclusion</h1>', unsafe_allow_html=True)
    
    p1 = """
        My computer vision project represents a significant contribution to how we use technology to interact 
        with the world around us, especially in the fields of road traffic and road safety. You have 
        demonstrated how smart cameras and computer vision systems can play a crucial role in monitoring 
        and optimizing road traffic, thereby contributing to traffic flow improvement.
        """
    transform(st, p1)

    p1 = """
        Road safety is a major concern in our societies, and my work has shown how computer vision can be used 
        to detect traffic violations, improve traffic light management and contribute to safer driving environments. 
        My efforts in this area have the potential to save lives especially from road accidents and by increasing 
        safety
        """
    transform(st, p1)

    p1 = """ 
        Additionally, by helping to ease traffic flow, I have paved the way for more effective traffic 
        management systems, including tackling congestion and improving the efficiency of our transportation 
        networks. My work illustrates the power of smart cameras and computer vision to solve complex and 
        challenging problems, contributing to smarter, more livable cities.
        """
    transform(st, p1)

    for i in range(10):
        transform(st, '')
    
    st.markdown(f'<a href="" target="_blank"><img src="{links()}" width="400" height="500"></a>', unsafe_allow_html=True)

    st.write('<h1 class="custom-text">Thanks for taking the time to read</h1>', unsafe_allow_html=True)

def styles():
     
    custom_css_title = """
        .header-text {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 25px; /* Taille de police */
            font-weight: bolder; /* Gras */
            text-decoration: underline; /* Souligné underline overline */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
        }

        .body-text {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 22px; /* Taille de police */
            font-weight: bolder; /* Gras */
            margin: 5px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
        }

        .header-text-under {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 15px; /* Taille de police */
            text-decoration: None; /* Souligné underline overline */
            margin: 10px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
        }

        .iou_schema {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 15px; /* Taille de police */
            text-decoration: None; /* Souligné underline overline */
            margin: 10px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
            font-family: Arial, sans-serif; /* font family*/
        }

    """
     
    return custom_css_title

def transform(st, text):
    s = text
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

def links():
    s = 'https://i.pinimg.com/originals/18/e1/11/18e1110635dc82318910603571fe4e5a.jpg'
    return s