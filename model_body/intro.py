def intro(st):
    st.write('<style>{}</style>'.format(styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Introduction</h1>', unsafe_allow_html=True)

    
    p1 = """
        In an era marked by technological advancements, the integration of computer vision and
        artificial intelligence has opened up new avenues for addressing critical societal issues. 
        This project, titled "Enhancing Road Safety through computer vision and optical character recognition 
        of license plates," represents a groundbreaking endeavor in the realm of traffic management and road safety. 
        The primary objective is to harness the power of smart cameras and the YOLO (You Only Look Once) 
        model to prevent traffic accidents and improve road safety, and with the optical character recognition 
        system of license plates, we can more effectively enhance the authorization of vehicles to circulate in certain areas.
        """
    
    p2 =""" 
        In a world where road traffic accidents pose a significant threat to human lives and 
        economic stability, the need for innovative solutions is more pressing than ever. 
        This project capitalizes on the potential of computer vision to revolutionize the 
        way we control traffic signals, enabling real-time analysis and decision-making to 
        mitigate risks and improve the efficiency of road networks. Through a comprehensive 
        approach encompassing data collection, model training, real-time detection, decision-making, 
        and advanced reporting mechanisms, this project seeks to create an intelligent traffic 
        control system that goes beyond traditional traffic management.
        """
    p3 = """ 
        By detecting vehicles, pedestrians, and traffic lights in real time and 
        dynamically adapting traffic signal timings, this system aims to prevent 
        accidents, reduce traffic congestion, and contribute to a safer and more 
        efficient road infrastructure, all while utilizing the optical character 
        recognition of license plates for more precise management of authorized 
        vehicles in specific areas.
        """

    p4 = """ 
        This project is not only a testament to the power of cutting-edge technology but also 
        a commitment to the safety and well-being of individuals on the road. By leveraging 
        computer vision, it offers a promising solution to one of 
        society's most pressing challenges—road safety enhancement in the face of ever-increasing traffic demands.
        """
    
    for p in [p1, p2, p3, p4]:
        transform(st, p)

def transform(st, text):
    s = text
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

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

    """
     
     return custom_css_title