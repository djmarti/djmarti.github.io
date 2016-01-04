#include "colors.inc"

#declare rd=pi/180;
#if (frame_number <=1000)
    #declare dist= 50 + 10*cos(1.95*pi*clock);
    #declare theta=-70*rd + 2*pi*clock;
    #declare phi=70*rd*cos(2*pi*clock);
#end
#if (frame_number > 1000)
    #declare dist= 50 + 10*cos(1.95*pi);
    #declare theta=-70*rd + pi*((clock - 0.833) + 0.167 * sin(pi*(clock-0.833)/0.167)/pi); // Smooth brake
    #declare phi=70*rd;
#end


camera {
    location <dist*sin(phi)*cos(theta), dist*sin(phi)*sin(theta), dist*cos(phi)>
    sky      <0,0,1>
    look_at  <0,0,25>
    angle    60
}
light_source { <30,30,50> color White}
light_source { <30,-30,50> color White}
light_source { <-30,30,50> color White} 
light_source { <-30,-30,50> color White} // right wing

background { color rgb <0,0,0> }

#declare crvTex=texture { pigment { Red } }
#include "attractor.inc"
#include concat ("cloud", str(frame_number,-4,0), ".inc")
