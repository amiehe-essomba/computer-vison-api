def sidebar_styles():
    # Définir le style CSS personnalisé pour la barre latérale
    custom_sidebar_style = """
    .sidebar-text {
        background-image: linear-gradient(to bottom, blue, skyblue, deepskyblue);
        color: rgb(0,0,0); /* Couleur du texte de la barre latérale */
        font-weight: bolder; /* Gras */
        text-align: left; /* Alignement du texte */
        text-transform: uppercase; /* uppercase */
        text-decoration: none; /* Souligné underline overline */ 
        /*display: inline-block;*/
        height: 35px; /* longueur */
        box-shadow: 5px 5px 10px 0 rgba(0, 0, 0.5, 5); /* Ombre */
        font-family: Arial, sans-serif; /* font family*/
        font-size: 1px; /* Taille de police */
        border-radius: 5px; /* Coins arrondis */
        margin: 3px; /* Marge extérieure */
        border: 2px solid #555; /* Bordure */
        padding: 5px; /* Marge intérieure pour le texte */
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
        font-weight: bolder; /* Gras */
        text-transform: capitalize; /* uppercase */
        font-weight: lighter; /* Gras */
        display: inline-block;
        border-radius: 5px; /* Coins arrondis */
        font-family: Arial, sans-serif; /* font family*/
        }

    .author-info {
        color: white; /* Couleur du texte */
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
        width:  320px; /* Largeur du conteneur */
        height: 150px; /* longueur */
        line-height: 1.5; /* Hauteur de ligne */
        font-family: Arial, sans-serif; /* font family*/
        text-align: justify;
        background-image: linear-gradient(to bottom, rgb(0,0,0), rgb(30, 30, 30), gray);
        }
    }
    """

    return custom_sidebar_style