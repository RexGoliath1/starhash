High level star simulation using Stelllarium 

Objectives: 
    High Accuracy Sim: 
        HIP Catalog (Gaia TODO)
        Projection (Stereographic)
        
    Mid Accurate Sim: 
        Visible Light (0.4 - 1.0 um)
        Optical effects (Haloing, Blur)

    Low Accuracy:
        Stray light effects

What sim does:
    Render star field
    Provide truth attitude/position
    Screen Capture (Python MSS)
    Broadcast screen capture UDP stream (ICP Socket) 
        TODO: RTSP or better non-compressed video codec

What sim doesn't:
    V&V level qualification (TODO: NASA GIANT investigation)
    HWIL Accuracy (magnitude matching and optical collimators required)
