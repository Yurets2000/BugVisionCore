from app import init_app

if __name__ == '__main__':
    init_app().run(host='0.0.0.0', port=5000, debug=True)
