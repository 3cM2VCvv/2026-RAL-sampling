"""Utilities for converting Buzzracer XML configs to dictionaries.

This module has no dependency on sacred — it is pure XML parsing.
"""
from xml.dom import minidom


def parse_xml_value(raw: str):
    """Parse an XML attribute value using the same eval-with-fallback
    pattern as ConfigObject.__init__ and Extension.load."""
    try:
        return eval(raw)
    except (NameError, SyntaxError):
        return raw


def xml_config_to_dict(config_filename: str) -> dict:
    """Parse a Buzzracer XML config file into a nested dict suitable for
    sacred's config system.

    Returns a dict with keys: settings, track, extensions, cars.
    """
    config = minidom.parse(config_filename)

    result = {}

    # Settings
    config_settings = config.getElementsByTagName('settings')[0]
    settings = {}
    for key, raw in config_settings.attributes.items():
        settings[key] = parse_xml_value(raw)
    # experiment_type is a child text node, not an attribute
    exp_type_nodes = config_settings.getElementsByTagName('experiment_type')
    if exp_type_nodes:
        settings['experiment_type'] = exp_type_nodes[0].firstChild.nodeValue
    result['settings'] = settings

    # Track
    track_nodes = config.getElementsByTagName('track')
    if track_nodes and track_nodes[0].firstChild:
        result['track'] = track_nodes[0].firstChild.nodeValue.strip()

    # Extensions
    extensions = {}
    ext_container = config.getElementsByTagName('extensions')
    if ext_container:
        for ext_node in ext_container[0].getElementsByTagName('extension'):
            class_name = ext_node.firstChild.nodeValue.strip()
            handle = ext_node.getAttribute('handle')
            ext_dict = {'class': class_name}
            for key, raw in ext_node.attributes.items():
                if key == 'handle':
                    continue
                ext_dict[key] = parse_xml_value(raw)
            extensions[handle] = ext_dict
    result['extensions'] = extensions

    # Cars
    cars = []
    cars_container = config.getElementsByTagName('cars')
    if cars_container:
        for car_node in cars_container[0].getElementsByTagName('car'):
            car_dict = {}
            # Simple child text elements
            for tag in ('name', 'hardware', 'config_name'):
                nodes = car_node.getElementsByTagName(tag)
                if nodes and nodes[0].firstChild:
                    car_dict[tag] = nodes[0].firstChild.nodeValue.strip()

            # Init states
            init_nodes = car_node.getElementsByTagName('init_states')
            if init_nodes and init_nodes[0].firstChild:
                car_dict['init_states'] = parse_xml_value(
                    init_nodes[0].firstChild.nodeValue.strip())

            # Controller
            ctrl_nodes = car_node.getElementsByTagName('controller')
            if ctrl_nodes:
                ctrl_node = ctrl_nodes[0]
                ctrl_dict = {}
                type_nodes = ctrl_node.getElementsByTagName('type')
                if type_nodes and type_nodes[0].firstChild:
                    ctrl_dict['type'] = type_nodes[0].firstChild.nodeValue.strip()
                for key, raw in ctrl_node.attributes.items():
                    ctrl_dict[key] = parse_xml_value(raw)
                car_dict['controller'] = ctrl_dict

            cars.append(car_dict)
    result['cars'] = cars

    return result
