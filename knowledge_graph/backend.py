from flask import Flask, jsonify, request
from py2neo import Graph

app = Flask(__name__)

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))


@app.route('/')
def index():
    return "Welcome to the Church History Knowledge Graph!"


@app.route('/persons', methods=['GET'])
def get_persons():
    query = """
    MATCH (p:Person)
    RETURN person
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/persons', methods=['GET'])
def get_persons():
    query = """
    MATCH (person:Person)
    RETURN person
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/places', methods=['GET'])
def get_places():
    query = """
    MATCH (place:Place)
    RETURN place
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/things', methods=['GET'])
def get_things():
    query = """
    MATCH (thing:Thing)
    RETURN thing
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/artifacts', methods=['GET'])
def get_artifacts():
    query = """
    MATCH (artifact:Artifact)
    RETURN artifact
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/denoms', methods=['GET'])
def get_denoms():
    query = """
    MATCH (denom:Denom)
    RETURN denom
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/denomgroups', methods=['GET'])
def get_denomgroups():
    query = """
    MATCH (denomgroup:DenomGroup)
    RETURN denomgroup
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/events', methods=['GET'])
def get_events():
    query = """
    MATCH (event:Event)
    RETURN event
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/writings', methods=['GET'])
def get_writings():
    query = """
    MATCH (writing:Writing)
    RETURN writing
    """
    data = graph.run(query).data()
    return jsonify(data)


@app.route('/concepts', methods=['GET'])
def get_concepts():
    query = """
    MATCH (concept:Concept)
    RETURN concept
    """
    data = graph.run(query).data()
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)