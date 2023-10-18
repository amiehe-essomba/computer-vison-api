def sidebar_styles():
    # Définir le style CSS personnalisé pour la barre latérale
    custom_sidebar_style = """
    .sidebar-text {
        color: black; /* Couleur du texte de la barre latérale */
        font-weight: bolder; /* Gras */
        text-align: left; /* Alignement du texte */
        text-transform: uppercase; /* uppercase */
        text-decoration: underline; /* Souligné underline overline */
        }

    .sub-sidebar-text {
        color: blue; /* color */
        text-align: left; /* Alignement du texte */
        margin: 10px; /* Marge extérieure */
        text-transform: capitalize; /* uppercase */
        }

    .author {
        color: black; /* color */
        text-align: left; /* Alignement du texte */
        text-transform: capitalize; /* uppercase */
        font-weight: lighter; /* Gras */
        }

    .author-info {
        color: black; /* Couleur du texte */
        background-color: white; /* Couleur de l'arrière-plan */
        padding: 5px; /* Marge intérieure pour le texte */
        border-radius: 5px; /* Coins arrondis */
        font-size: 13px; /* Taille de police */
        font-weight: lighter; /* Gras */
        text-align: left; /* Alignement du texte */
        text-transform: proper; /* Texte en majuscules capitalize, lowercase*/
        text-decoration: none; /* Souligné underline overline */
        box-shadow: 2px 2px 5px 0 rgba(0, 0, 0, 0.2); /* Ombre */
        border: 2px solid #555; /* Bordure */
        margin: 0px; /* Marge extérieure */
        width:  300px; /* Largeur du conteneur */
        height: 170px; /* longueur */
        line-height: 1.5; /* Hauteur de ligne */
        font-family: Arial, sans-serif; /* font family*/
        text-align: justify;
        }
    }
    """

    return custom_sidebar_style