import rdflib


class PMDco:
    ns = rdflib.Namespace("https://w3id.org/pmd/co/PMD_")

    atomic_structure = ns["0000526"]
    bulk_modulus = ns["0000539"]
    chemical_composition = ns["0000551"]
    energy = ns["0020142"]
    gibbs_energy = ns["0020361"]
    length = ns["0040001"]
    pressure = ns["0000896"]
    volume = ns["0020150"]


class AMSO:
    ns = rdflib.Namespace("https://purls.helmholtz-metadaten.de/asmo/")

    strain = ns["Strain"]


class URI:
    atomic_structure = PMDco.atomic_structure
    bulk_modulus = PMDco.bulk_modulus
    chemical_symbol = PMDco.chemical_composition
    energy = PMDco.energy
    gibbs_energy = PMDco.gibbs_energy
    length = PMDco.length
    pressure = PMDco.pressure
    strain = AMSO.strain
    volume = PMDco.volume
