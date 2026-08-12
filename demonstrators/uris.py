import rdflib


class PMDco:
    ns = rdflib.Namespace("https://w3id.org/pmd/co/PMD_")

    atomic_structure = ns["0000526"]
    bulk = ns["0000538"]
    bulk_modulus = ns["0000539"]
    chemical_composition = ns["0000551"]
    energy = ns["0020142"]
    three_d = ns["0025005"]
