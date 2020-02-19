#version 130

out vec4 gl_FragColor;

const vec2 iResolution = vec2(1920, 1080);
const vec3 c = vec3(1.,0.,-1.);

uniform float iProgress;

void dlinesegment(in vec2 x, in vec2 p1, in vec2 p2, out float d)
{
    vec2 da = p2-p1;
    d = length(x-mix(p1, p2, clamp(dot(x-p1, da)/dot(da,da),0.,1.)));
}

float sm(in float d)
{
    return smoothstep(1.5/iResolution.y, -1.5/iResolution.y, d);
}

void main()
{
    vec2 uv = (gl_FragCoord.xy-.5*iResolution.xy)/iResolution.y;
    vec3 col = c.yyy;
	float d;
    
    // Outside
    dlinesegment(uv, -.4*c.xy, .4*c.xy, d);
    d = abs(d-.05)-.005;
    col = mix(col, c.xxx, sm(d));
    d = abs(d-.006)-.001;
    col = mix(col, .5*c.xxx, sm(d));
    
    // Progress
    dlinesegment(uv, -.4*c.xy, (.8*iProgress-.4)*c.xy, d);
    d -= .04;
    col = mix(col, mix(.7,1.,uv.y/.05)*c.xxx, sm(d));
    
    gl_FragColor = vec4(clamp(col,0.,1.),1.0);
}
