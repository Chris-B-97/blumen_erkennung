from modell import load_model, predict_image

def main():
    model = load_model()

    print("Willkommen im Blumen-Erkennungsprogramm!")
    print("Bitte geben Sie den Pfad zum Bild ein (z. B.: 'bilder/rose.jpg'): ")

    image_path = input()
    predict_image(image_path, model)

if __name__ == "__main__":
    main()
