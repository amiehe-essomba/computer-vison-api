def font(path : str = './font/'):
    import matplotlib.font_manager

    font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    for i, font_path in enumerate(font_list):
        name = font_path.split("\\")[-1].lower()

        if name.split(".")[0] in ["arial", 'arialbd', "calibri", "calibril", 
                                        'consolai', "consolab", "calibriz", "corbell", 'micross']:
            
            try:
                with open(font_path, "rb") as font_file:
                    font_content = font_file.read()
              
                # Sauvegarder le contenu dans un nouveau fichier
                with open(f"./{path}/{name}", "wb") as output_file:
                    output_file.write(font_content)

                print(f"Le fichier de police a été sauvegardé avec succès dans ./font/{name}")
            except FileNotFoundError:
                print(f"Le fichier de police '{font_path}' n'a pas été trouvé.")
            except Exception as e:
                print(f"Une erreur s'est produite : {e}")