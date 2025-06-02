from tkinter import *
from tkinter import ttk
from functools import partial
#global settings
settings = {"render_DPI": {"value":200,
                           "changeable":True,
                           "type":"String",
                           "display_name":"Render DPI",
                           "entry_field":None,
                           "label_field":None

                        },
            "render_layer_height": {"value":0.1,
                           "changeable":True,
                            "type":"String",
                           "display_name":"Render Layer Height",
                           "entry_field":None,
                           "label_field":None

                        },
            "slice_DPI": {"value":975,
                           "changeable":True,
                            "type":"String",
                           "display_name":"Slice DPI",
                            "entry_field":None,
                            "label_field":None
                        },
            "slice_layer_height": {"value":0.005,
                           "changeable":True,
                            "type":"String",
                           "display_name":"Slice Layer Height",
                           "entry_field":None,
                           "label_field":None

                        },
            "support_generation": {"value":False,
                           "changeable":True,
                            "type":"Check",
                           "display_name":"Support Generation",
                           "entry_field":None,
                           "label_field":None

                        },
            "support_spacing": {"value":4,
                           "changeable":True,
                            "type":"String",
                           "display_name":"Support Structural Spacing",
                           "entry_field":None,
                           "label_field":None

                        },
            "traces_slicing": {"value":True,
                           "changeable":True,
                            "type":"Check",
                           "display_name":"Slice Traces",
                           "entry_field":None,
                           "label_field":None

                        },
"trace_width": {"value":0.5,
                           "changeable":True,
                            "type":"String",
                           "display_name":"Circuit Trace Width",
                           "entry_field":None,
                           "label_field":None

                        },
            "traces_print_height": {"value":4,
                           "changeable":True,
                            "type":"String",
                           "display_name":"Printhead spacing",
                           "entry_field":None,
                           "label_field":None

                        },
            "printhead_UV_offset": {"value":30,
                                    "changeable": True,
                                    "type": "String",
                                    "display_name": "UV light offset",
                                    "entry_field": None,
                                    "label_field": None
                                    },
            "sec_struc_transparancy": {"value":20,
                                    "changeable": True,
                                    "type": "String",
                                    "display_name": "Secondary structure transparancy",
                                    "entry_field": None,
                                    "label_field": None
                                    },
            "render_components": {"value": True,
                                   "changeable": True,
                                   "type": "Check",
                                   "display_name": "Render electronic components",
                                   "entry_field": None,
                                   "label_field": None

                                   },
            }


def settings_menu():
    global settings
    global base

    entry_field_spacing = 40
    entry_field_width = 300
    entry_field_x0 = 20
    entry_field_y0 = 40
    label_text_width = 30


    settings_count = len(settings.keys())
    base = Tk()
    base.geometry(f"500x{(settings_count+2)*entry_field_spacing}")
    base.title("Slicer Settings")

    current_entry_field = 0

    for key in settings.keys():
        if(settings[key]["changeable"]):
            settings[key]["label_field"] = Label(base, text=settings[key]["display_name"], width=label_text_width,
                                                 font=("arial", 12))
            settings[key]["label_field"].place(x=0,
                                               y=entry_field_y0 + current_entry_field * entry_field_spacing)
            if(settings[key]["type"]=="String"):
                settings[key]["entry_field"] = Entry(base)
                settings[key]["entry_field"].place(x=entry_field_width, y=entry_field_y0 + current_entry_field * entry_field_spacing)
                settings[key]["entry_field"].insert(0,str(settings[key]["value"]))
            if(settings[key]["type"]=="Check"):
                settings[key]["entry_field"] = ttk.Checkbutton(base,takefocus = 0)
                settings[key]["entry_field"].place(x=entry_field_width, y=entry_field_y0 + current_entry_field * entry_field_spacing)
                settings[key]["entry_field"].state(['!alternate'])
                if(settings[key]["value"]):
                    settings[key]["entry_field"].state(['selected'])


                #settings[key]["entry_field"].insert(0,str(settings[key]["value"]))

            current_entry_field += 1


    Button(base, text="Save", width=10, command=save_settings).place(x=entry_field_x0, y=entry_field_y0 +current_entry_field * entry_field_spacing)
    Button(base, text="Cancel", width=10, command=cancel_settings).place(x=entry_field_width, y=entry_field_y0 +current_entry_field * entry_field_spacing)
    base.mainloop()


def cancel_settings():
    global base
    base.destroy()
    return

def save_settings():
    global base
    global settings

    for key in settings.keys():
        if(settings[key]["changeable"]):
            if(settings[key]["type"]=="String"):
                settings[key]["value"] = settings[key]["entry_field"].get()
            if (settings[key]["type"] == "Check"):
                if(settings[key]["entry_field"].state()==("selected",)):
                    settings[key]["value"] = True
                else:
                    settings[key]["value"] = False

    base.destroy()
    #print(settings)
    return


def tracer_menu(circuit_layers,active_circuit_layer_passed):
    field_spacing = 40
    field_y0 = 20
    global program_state
    program_state = "slicer"
    global active_circuit_layer
    active_circuit_layer = active_circuit_layer_passed

    circuit_layers = sorted(circuit_layers, key=lambda x: x.z_height)
    #print(circuit_layers)
    base = Tk()
    base.geometry(f"500x{(len(circuit_layers) + 3) * field_spacing}")
    base.title("Circuit layers")
    for i,layer in enumerate(circuit_layers):
        Label(base, text=f"z={layer.z_height:.2f}", width=10,font=("arial", 12)).place(x=10,y=i*field_spacing+field_y0)
        Button(base, text="Edit", width=20,command=partial(edit_circuit_layer,layer.z_height,circuit_layers,base)).place(x=100,y=i*field_spacing+field_y0)
        Button(base, text="Remove", width=20,command=partial(remove_circuit_layer,layer.z_height,circuit_layers,base)).place(x=300, y=i * field_spacing + field_y0)
    Button(base, text="Cancel", width=10, command=lambda: base.destroy()).place(x=10, y=(len(circuit_layers) + 1) * field_spacing + field_y0)
    base.mainloop()
    return circuit_layers,program_state,active_circuit_layer

def edit_circuit_layer(z_height,circuit_layers,base):
    global active_circuit_layer
    for i,layer in enumerate(circuit_layers):
        if(layer.z_height == z_height):
            active_circuit_layer = circuit_layers[i]
    global program_state
    program_state = "tracer"
    base.destroy()
    return


def remove_circuit_layer(z_height,circuit_layers,base):
    field_spacing = 40
    field_y0 = 20
    for i,layer in enumerate(circuit_layers):
        if(layer.z_height == z_height):
            circuit_layers.pop(i)
            print(z_height)
    for widget in base.winfo_children():
        widget.destroy()
    for i,layer in enumerate(circuit_layers):
        Label(base, text=f"z={layer.z_height:.2f}", width=10,font=("arial", 12)).place(x=10,y=i*field_spacing+field_y0)
        Button(base, text="Edit", width=20,command=partial(edit_circuit_layer,layer.z_height,circuit_layers,base)).place(x=100,y=i*field_spacing+field_y0)
        Button(base, text="Remove", width=20,command=partial(remove_circuit_layer,layer.z_height,circuit_layers,base)).place(x=300, y=i * field_spacing + field_y0)
    Button(base, text="Cancel", width=10, command=lambda: base.destroy()).place(x=10, y=(len(circuit_layers) + 1) * field_spacing + field_y0)
    return