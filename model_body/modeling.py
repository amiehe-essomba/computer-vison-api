def modeling(st):
    import streamlit as st

    st.write('<style>{}</style>'.format(styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Data modelling and understanding</h1>', unsafe_allow_html=True)
    
    st.write(f'<h1 class="body-text">Inputs and outputs</h1>', unsafe_allow_html=True)
    p1 = """
        The input is a batch of images, and each image has the shape (608, 608, 3)
        The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented
         by 6 numbers (pc, bx, by, bh, bw, c). If you expand c
        into an 80-dimensional vector, each bounding box is then represented by 85 numbers [[pc, bx, by, bh, bw, c], [80 class probabilities]].
        """
    transform(st, p1)

    st.write(f'<h1 class="body-text">Anchor Boxes</h1>', unsafe_allow_html=True)
    p2 = """
        Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that 
        represent the different classes. For this assignment, 5 anchor boxes have been chosen (to cover the 80 classes)
        The dimension of the encoding tensor of the second to last dimension based on the anchor boxes is  
        (𝑚, nH, nW, anchors, classes).
        The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).
        """
    transform(st, p2)

    st.write(f'<h1 class="body-text">Class Score</h1>', unsafe_allow_html=True)

    t=r'''
    \begin{align}
        scores = pc \times 
        \begin{pmatrix}
            1. \\
            1. \\
            1. \\
            \vdots \\
            1. \\
            1. \\
            1.
        \end{pmatrix} \times
        \begin{pmatrix}
            c2 \\
            c1 \\
            c3 \\
            \vdots \\
            c78 \\
            c79 \\
            c80
        \end{pmatrix} = 
        \begin{pmatrix}
            pc \times c2 \\
            pc \times c1 \\
            pc \times c3 \\
            \vdots \\
            pc \times c78 \\
            pc \times c79 \\
            pc \times c80
        \end{pmatrix} 
    \end{align}
    '''
    st.latex(t)

    st.latex(
    r'''
    \begin{equation}
        class = argmax(scores)
    \end{equation}
    '''
    )

    st.write(f'<h1 class="body-text">Yolo Filter Boxes</h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="body-text">IoU</h1>', unsafe_allow_html=True)
    st.write(f'<h1 class="body-text">Non-Max Suppression</h1>', unsafe_allow_html=True)

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
        }

        .body-text {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 22px; /* Taille de police */
            font-weight: bolder; /* Gras */
            margin: 5px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
        }

        .header-text-under {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 20px; /* Taille de police */
            text-decoration: None; /* Souligné underline overline */
            margin: 10px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
        }

    """
     
     return custom_css_title