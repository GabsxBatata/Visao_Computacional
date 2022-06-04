##base de dados para reconhecer rostos
import face_recognition as fr


def reconhece_face(url_foto):
    #carregar foto
    foto = fr.load_image_file(url_foto)
    ##analisando para reconhecer face
    rostos = fr.face_encodings(foto)
    ##se achar rosto
    if(len(rostos) > 0):
        return True, rostos

    return False, []


def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []
##teste para indentificar rosto Carlos Gabriel
    CarlosGabriel = reconhece_face("./img/CarlosGabriel.jpeg")
    if(CarlosGabriel[0]):
##definição de teste do rosto na imagem
        rostos_conhecidos.append(CarlosGabriel[1][0])
        nomes_dos_rostos.append("Carlos Gabriel")
##teste para indentificar face Douglas
    Douglas = reconhece_face("./img/Douglas.jpeg")
    if(Douglas[0]):
##definição de teste do rosto imagem
        rostos_conhecidos.append(Douglas[1][0])
        nomes_dos_rostos.append("Douglas")

    return rostos_conhecidos, nomes_dos_rostos
