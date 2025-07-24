from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        "message": "YOLOv8 Web App is running!",
        "status": "success",
        "deployment": "Railway"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 