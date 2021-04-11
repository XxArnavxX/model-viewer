from flask import Flask, request, jsonify
from classifiar import getimage
app = Flask(__name__)
@app.route("/predict-digit", methods = ["POST"])
def predictdata():
    image = request.files.get("digit")
    predictions = getimage(image)
    return jsonify({
        "prediction": predictions,
    })


if(__name__=="__main__"):
    app.run(debug=True)