import os
import tempfile


import pytest


from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client



def test_base_endpoint_get(client):
    """" test base endpoint """

    response = client.get('/')
    assert response.status_code == 404


def test_base_endpoint_post(client):
    response = client.post('/')
    assert response.status_code == 404


def test_predict_endpoint_get(client):
    """" test predict endpoint """

    # test get
    response = client.get('/predict')
    assert response.status_code == 405


def test_predict_endpoint_post_nojson(client):
    # test invalid post
    response = client.post('/predict')
    assert response.status_code == 400


def test_predict_endpoint_post_valid_json(client):
    # test valid post
    response = client.post('/predict', json={
        "model": "multinomial",
        "vectorizer": "count",
        "text": "it's a beautiful world"
    })
    assert response.status_code == 200
    assert response.data in [b"positive", b"negative"]
    assert isinstance(response.data, bytes)


def test_predict_all_endpoint_get(client):
    """ test predict endpoint """
    # test get
    response = client.get('/predict_all')
    assert response.status_code == 405


def test_predict_all_endpoint_unnecessary_args(client):
    # test post with unnecessary parameters
    response = client.post('/predict_all', json={
        "model": "multinomial",
        "vectorizer": "count",
        "text": "it's a beautiful world"
    })
    assert response.status_code == 400


def test_predict_all_endpoint_post_invalid_json_key(client):
    # test invalid json post
    response = client.post('/predict_all', json={
        "message": "it's a beautiful world"
    })
    assert response.status_code == 400


def test_predict_all_endpoint_invalid_json_value(client):
    # test invalid json value post
    response = client.post('/predict_all', json={
        "message": 1
    })
    assert response.status_code == 400


def test_predict_all_endpoint_valid_post(client):
    # test valid post
    response = client.post('/predict_all', json={
        "text": "it's a beautiful world"
    })
    assert response.status_code == 200
    print(response.json)
    assert response.json["multinomial"]["tfidf"] in ["positive", "negative"]
    assert isinstance(response.json, dict)


