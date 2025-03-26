icon_count = 0
icon_width = 50
icon_spacing = 20


def create_button(pg,window,icon_name,ui_batch,func,reset=False):
    global icon_count
    global icon_width
    global icon_spacing
    if(reset): icon_count=0
    img = pg.resource.image(icon_name)
    button = pg.gui.PushButton(x=icon_spacing+icon_count*(icon_width+icon_spacing), y=icon_spacing, pressed=img, depressed=img, batch=ui_batch)
    window.push_handlers(button)
    icon_count+=1
    button.set_handler('on_press', func)
    return button

