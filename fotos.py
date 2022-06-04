import face_recognition as fr
from engine import reconhece_face, get_rostos
#reconhecendo pessoa na foto
desconhecido = reconhece_face("./img/desconhecido.jpeg")
if(desconhecido[0]):
    rosto_desconhecido = desconhecido[1][0]
    rosto_conhecidos, nomes_dos_rostos = get_rostos()
    resultados = fr.compare_faces(rosto_conhecidos, rosto_desconhecido)
    print(resultados)

#regra para se for encontrado rosto ou não
    for i in range(len(rosto_conhecidos)):
        resultado = resultados[i]
        if(resultado):
            print("Rosto do", nomes_dos_rostos[i], "foi reconhecido")

else:
    print("Não foi encontrado nenhum rosto")
