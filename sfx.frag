#version 130
const float PI = radians(180.);
const float TAU = 2.*PI;
float clip(float a) { return clamp(a,-1.,1.); }
float smstep(float a, float b, float x) {return smoothstep(a, b, clamp(x, a, b));}
float theta(float x) { return smstep(0.,1e-3,x); }
float _sin(float a) { return sin(TAU * mod(a,1.)); }
float _sin_(float a, float p) { return sin(TAU * mod(a,1.) + p); }
float _sq_(float a,float pwm) { return sign(2.*fract(a) - 1. + pwm); }
float _tri(float a) { return (4.*abs(fract(a)-.5) - 1.); }
float freqC1(float note){ return 32.7 * exp2(note/12.); }
float minus1hochN(int n) { return (1. - 2.*float(n % 2)); }
float minus1hochNminus1halbe(int n) { return sin(.5*PI*float(n)); }
float pseudorandom(float x) { return fract(sin(dot(vec2(x),vec2(12.9898,78.233))) * 43758.5453); }
float fhelp(float x) { return 1. + .333*x; } // 1. + .33333*x + .1*x*x + .02381*x*x*x + .00463*x*x*x*x;
float linmix(float x, float a, float b, float y0, float y1) { return mix(y0,y1,clamp(a*x+b,0.,1.)); }
float s_atan(float a) { return .636 * atan(a); }

#define NTIME 2
const float pos_B[2] = float[2](0.,56.);
const float pos_t[2] = float[2](0.,98.8235);
const float pos_BPS[1] = float[1](.5667);
const float pos_SPB[1] = float[1](1.7646);
float BPS, SPB, BT;

float Tsample;

#define filterthreshold 1.e-3

//TEXCODE

float drop_phase(float time, float t1, float f0, float f1)
{
    float t = min(time, t1);
    float phi = f0*t + .5*(f1-f0)/t1*t*t;

    if(time > t1)
    {
        phi += f1 * (time - t1);
    }
    return phi;
}

float metalnoise(float t, float fac1, float fac2)
{
    return .666*pseudorandom(t) - 1.333*pseudorandom(t-Tsample) + .333*(pseudorandom(t+fac1*Tsample)+pseudorandom(t+fac2*Tsample));
}

float lpnoise(float t, float fq)
{
    t *= fq;
    float tt = fract(t);
    float tn = t - tt;
    return mix(pseudorandom(floor(tn) / fq), pseudorandom(floor(tn + 1.0) / fq), smstep(0.0, 1.0, tt));
}

float reverb_phase(float t, float amt)
{
    float r = lpnoise(t, 100.0) + 0.2*lpnoise(t, 550.0) + 0.1*lpnoise(t, 1050.0)*exp(-5.*t);
    return amt * r;
}

float env_AHDSR(float x, float L, float A, float H, float D, float S, float R)
{
    return (x<A ? x/A : x<A+H ? 1. : x<A+H+D ? (1. - (1.-S)*(x-H-A)/D) : x<=L-R ? S : x<=L ? S*(L-x)/R : 0.);
}

float env_limit_length(float x, float length, float release)
{
    return clamp(x * 1e3, 0., 1.) * clamp(1 - (x-length)/release, 0., 1.);
}

float waveshape(float s, float amt, float A, float B, float C, float D, float E)
{
    float w;
    float m = sign(s);
    s = abs(s);

    if(s<A) w = B * smstep(0.,A,s);
    else if(s<C) w = C + (B-C) * smstep(C,A,s);
    else if(s<=D) w = s;
    else if(s<=1.)
    {
        float _s = (s-D)/(1.-D);
        w = D + (E-D) * (1.5*_s*(1.-.33*_s*_s));
    }
    else return 1.;

    return m*mix(s,w,amt);
}

float sinshape(float x, float amt, float parts)
{
    return (1.-amt) * x + amt * sign(x) * 0.5 * (1. - cos(parts*PI*x));
}

float comp_SAW(int N, float inv_N, float PW) {return inv_N * (1. - _sin(float(N)*PW));}
float comp_TRI(int N, float inv_N, float PW) {return N % 2 == 0 ? .1 * inv_N * _sin(float(N)*PW) : inv_N * inv_N * (1. - _sin(float(N)*PW));}
float comp_SQU(int N, float inv_N, float PW) {return inv_N * (minus1hochN(N) * _sin(.5*float(N)*PW + .25) - 1.);}
float comp_HAE(int N, float inv_N, float PW) {return N % 2 == 0 ? 0. : inv_N * (1. - minus1hochNminus1halbe(N))*_sin(PW);}

float MADD(float t, float f, float p0, int NMAX, int NINC, float MIX, float CO, float NDECAY, float RES, float RES_Q, float DET, float PW, float LOWCUT, int keyF)
{
    float ret = 0.;
    float INR = keyF==1 ? 1./CO : f/CO;
    float IRESQ = keyF==1 ? 1./RES_Q : 1./(RES_Q*f);

    float p = f*t;
    float float_N, inv_N, comp_mix, filter_N;
    for(int N = 1 + int(LOWCUT/f - 1.e-3); N<=NMAX; N+=NINC)
    {
        float_N = float(N);
        inv_N = 1./float_N;
        comp_mix = MIX < 0. ? (MIX+1.) * comp_TRI(N,inv_N,PW)  -     MIX  * comp_SAW(N,inv_N,PW)
                 : MIX < 1. ? (1.-MIX) * comp_TRI(N,inv_N,PW)  +     MIX  * comp_SQU(N,inv_N,PW)
                            : (MIX-1.) * comp_HAE(N,inv_N,PW)  + (2.-MIX) * comp_SQU(N,inv_N,PW);

        if(abs(comp_mix) < 1e-6) continue;

        filter_N = pow(1. + pow(float_N*INR,NDECAY),-.5) + RES * exp(-pow((float_N*f-CO)*IRESQ,2.));

        ret += comp_mix * filter_N * (_sin_(float_N * p, p0) + _sin_(float_N * p * (1.+DET), p0));
    }
    return s_atan(ret);
}

float MADD(float t, float f, float p0, int NMAX, int NINC, float MIX, float CO, float NDECAY, float RES, float RES_Q, float DET, float PW, int keyF)
{
    return MADD(t, f, p0, NMAX, NINC, MIX, CO, NDECAY, RES, RES_Q, DET, PW, 0., keyF);
}

float QFM_FB(float PH, float FB) // my guessing of feedback coefficients, FB>0 'saw', FB<0 'sq'
{
    if(FB > 0.) return abs(FB) * .8*_sin(PH + .35*_sin(PH));
    else return abs(FB) * _sin(PH + .5*PI);
}

float QFM(float t, float f, float phase, float LV1, float LV2, float LV3, float LV4, float FR1, float FR2, float FR3, float FR4, float FB1, float FB2, float FB3, float FB4, float ALGO)
{
    int iALGO = int(ALGO);
    float PH1 = FR1 * f * t + phase;
    float PH2 = FR2 * f * t + phase;
    float PH3 = FR3 * f * t + phase;
    float PH4 = FR4 * f * t + phase;

    float LINK41 = 0., LINK42 = 0., LINK43 = 0., LINK32 = 0., LINK31 = 0., LINK21 = 0.;
    if(iALGO == 1)       {LINK43 = 1.; LINK32 = 1.; LINK21 = 1.;}
    else if(iALGO == 2)  {LINK42 = 1.; LINK32 = 1.; LINK21 = 1.;}
    else if(iALGO == 3)  {LINK41 = 1.; LINK32 = 1.; LINK21 = 1.;}
    else if(iALGO == 4)  {LINK42 = 1.; LINK43 = 1.; LINK31 = 1.; LINK21 = 1.;}
    else if(iALGO == 5)  {LINK41 = 1.; LINK31 = 1.; LINK21 = 1.;}
    else if(iALGO == 6)  {LINK43 = 1.; LINK32 = 1.;}
    else if(iALGO == 7)  {LINK43 = 1.; LINK32 = 1.; LINK31 = 1.;}
    else if(iALGO == 8)  {LINK21 = 1.; LINK43 = 1.;}
    else if(iALGO == 9)  {LINK43 = 1.; LINK42 = 1.; LINK41 = 1.;}
    else if(iALGO == 10) {LINK43 = 1.; LINK42 = 1.;}
    else if(iALGO == 11) {LINK43 = 1.;}

    float OP4 = LV4 * _sin(PH4 + QFM_FB(PH4, FB4));
    float OP3 = LV3 * _sin(PH3 + QFM_FB(PH3, FB3) + LINK43*OP4);
    float OP2 = LV2 * _sin(PH2 + QFM_FB(PH2, FB2) + LINK42*OP4 + LINK32*OP3);
    float OP1 = LV1 * _sin(PH1 + QFM_FB(PH1, FB1) + LINK41*OP4 + LINK31*OP3 + LINK32*OP2);

    float sum = OP1;
    if(LINK21 > 0.) sum += OP2;
    if(LINK31 + LINK32 > 0.) sum += OP3;
    if(LINK41 + LINK42 + LINK43 > 0.) sum += OP4;

    return s_atan(sum);
}

float bandpassBPsaw01(float time, float f, float tL, float vel, float fcenter, float bw, float M)
{
    float y = 0.;
        
    float facM = 2.*PI/M;
    float facL = 2.*PI*Tsample * (fcenter - bw);
    float facH = 2.*PI*Tsample * (fcenter + bw);
    
    if(facL < 0.) facL = 0.;
    if(facH > PI) facH = PI;
    
    float _TIME, mm, w, h;
    
    M--;
    for(float m=1.; m<=M; m++)
    {
        mm = m - .5*M;
        w = .42 - .5 * cos(mm*facM) - .08 * cos(2.*mm*facM);
        h = 1./(PI*mm) * (sin(mm*facH) - sin(mm*facL));
        
        _TIME = time - m*Tsample;
        y += w*h*(2.*fract(f*_TIME)-1.);
    }
    
    return s_atan(M*M*y); // I DO NOT CARE ANYMORE
}

float protokick(float t, float f_start, float f_end, float fdecay, float hold, float decay, float drive, float detune, float rev_amount, float rev_hold, float rev_decay, float rev_drive)
{
    float phi = drop_phase(t, fdecay, f_start, f_end);
    float rev_phi = phi + reverb_phase(t, 1.);
    return clamp(drive*.5*(_sin(phi)+_sin((1.-detune)*phi)),-1.,1.) * exp(-max(t-hold, 0.)/decay)
         + rev_amount*clamp(rev_drive*.5*(_sin(rev_phi)+_sin((1.-detune)*rev_phi)),-1.,1.) * exp(-max(t-rev_hold, 0.)/rev_decay);
}

float _HCybHHClENV0(float t){return t <=.004? linmix(t,250.,0.,0.,1.):t <=.032? linmix(t,35.7143,-.1429,1.,0.):0.;}
float _HCybHHClENV1(float t){return t <=.009? linmix(t,111.1111,0.,0.,1.):t <=.06? linmix(t,19.6078,-.1765,1.,0.):0.;}
float _HCybHHClENV2(float t){return t <=.15? linmix(t,6.6667,0.,6000.,1000.):t <=.4? linmix(t,4.,-.6,1000.,200.):200.;}
float _HCybHHClENV3(float t){return t <=.18? linmix(t,5.5556,0.,4.123,4.04):4.04;}

uniform float iBlockOffset;
uniform float iSampleRate;
uniform float iTexSize;
uniform sampler2D iSequence;
uniform float iSequenceWidth;

// Read short value from texture at index off
float rshort(in float off)
{
    float hilo = mod(off, 2.);
    off = .5*off;
    vec2 ind = vec2(mod(off, iSequenceWidth), floor(off/iSequenceWidth));
    vec4 block = texelFetch(iSequence, ivec2(ind), 0);
    vec2 data = mix(block.rg, block.ba, hilo);
    return round(dot(vec2(255., 65280.), data));
}

// Read float value from texture at index off
float rfloat(int off)
{
    float d = rshort(float(off));
    float sign = floor(d/32768.),
        exponent = floor(d*9.765625e-4 - sign*32.),
        significand = d-sign*32768.-exponent*1024.;

    if(exponent == 0.)
         return mix(1., -1., sign) * 5.960464477539063e-08 * significand;
    return mix(1., -1., sign) * (1. + significand * 9.765625e-4) * pow(2.,exponent-15.);
}

#define NTRK 10
#define NMOD 201
#define NPTN 16
#define NNOT 280
#define NDRM 48

int trk_sep(int index)      {return int(rfloat(index));}
int trk_syn(int index)      {return int(rfloat(index+1+1*NTRK));}
float trk_norm(int index)   {return     rfloat(index+1+2*NTRK);}
float trk_rel(int index)    {return     rfloat(index+1+3*NTRK);}
float trk_pre(int index)    {return     rfloat(index+1+4*NTRK);}
float trk_slide(int index)  {return     rfloat(index+1+5*NTRK);} // idea for future: change to individual note_slide_time
float mod_on(int index)     {return     rfloat(index+1+6*NTRK);}
float mod_off(int index)    {return     rfloat(index+1+6*NTRK+1*NMOD);}
int mod_ptn(int index)      {return int(rfloat(index+1+6*NTRK+2*NMOD));}
float mod_transp(int index) {return     rfloat(index+1+6*NTRK+3*NMOD);}
int ptn_sep(int index)      {return int(rfloat(index+1+6*NTRK+4*NMOD));}
float note_on(int index)    {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN);}
float note_off(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+1*NNOT);}
float note_pitch(int index) {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+2*NNOT);}
float note_pan(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+3*NNOT);}
float note_vel(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+4*NNOT);}
float note_slide(int index) {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+5*NNOT);}
float note_aux(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+6*NNOT);}
float drum_rel(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+7*NNOT);}

vec2 mainSynth(float time)
{
    float sL = 0.;
    float sR = 0.;
    float dL = 0.;
    float dR = 0.;

    int _it;
    for(_it = 0; _it < NTIME - 2 && pos_t[_it + 1] < time; _it++);
    BPS = pos_BPS[_it];
    SPB = pos_SPB[_it];
    BT = pos_B[_it] + (time - pos_t[_it]) * BPS;

    float time2 = time - .0002;
    float sidechain = 1.;

    float amaysynL, amaysynR, amaydrumL, amaydrumR, B, Bon, Boff, Bprog, Bproc, L, tL, _t, _t2, vel, rel, pre, f, amtL, amtR, env, slide, aux;
    int tsep0, tsep1, _modU, _modL, ptn, psep0, psep1, _noteU, _noteL, syn, drum;

    for(int trk = 0; trk < NTRK; trk++)
    {
        tsep0 = trk_sep(trk);
        tsep1 = trk_sep(trk + 1);

        syn = trk_syn(trk);
        rel = trk_rel(trk);
        pre = trk_pre(trk);

        for(_modU = tsep0; (_modU < tsep1 - 1) && (BT > mod_on(_modU + 1) - pre); _modU++);
        for(_modL = tsep0; (_modL < tsep1 - 1) && (BT >= mod_off(_modL) + rel); _modL++);

        for(int _mod = _modL; _mod <= _modU; _mod++)
        {
            B = BT - mod_on(_mod);

            ptn   = mod_ptn(_mod);
            psep0 = ptn_sep(ptn);
            psep1 = ptn_sep(ptn + 1);

            for(_noteU = psep0; (_noteU < psep1 - 1) && (B > note_on(_noteU + 1) - pre); _noteU++);
            for(_noteL = psep0; (_noteL < psep1 - 1) && (B >= note_off(_noteL) + rel); _noteL++);

            for(int _note = _noteL; _note <= _noteU; _note++)
            {
                if(syn == 121)
                {
                    drum = int(note_pitch(_note));
                    rel = drum_rel(drum);
                }

                amaysynL  = 0.;
                amaysynR  = 0.;
                amaydrumL = 0.;
                amaydrumR = 0.;

                Bon   = note_on(_note) - pre;
                Boff  = note_off(_note) + rel;
                L     = Boff - Bon;
                tL    = L * SPB;
                Bprog = B - Bon;
                Bproc = Bprog / L;
                _t    = Bprog * SPB;
                _t2   = _t - .0002;
                vel   = note_vel(_note);
                amtL  = clamp(1. - note_pan(_note), 0., 1.);
                amtR  = clamp(1. + note_pan(_note), 0., 1.);
                slide = note_slide(_note);
                aux   = note_aux(_note);

                if(syn == 121)
                {
                    env = trk_norm(trk) * theta(Bprog) * theta(L - Bprog);
                    if(drum == 0) { sidechain = min(sidechain, 1. - vel * (clamp(1.e4 * Bprog,0.,1.) - pow(Bprog/(L-rel),8.)));}
                    else if(drum == 2){
                        amaydrumL = vel*vel*1.3*exp(-7.*max(_t-.05+10.*vel,0.))*metalnoise(.6*_t, .5, 2.)
      +vel*1.3*(lpnoise(_t,10000.)*smstep(0.,.01,_t)*(1.-(1.-.13)*smstep(0.,.12,_t-.01))-.3*(1.00*lpnoise((_t-0.00),10000.)*smstep(0.,.01,(_t-0.00))*(1.-(1.-.13)*smstep(0.,.12,(_t-0.00)-.01))+6.10e-01*lpnoise((_t-1.20e-03),10000.)*smstep(0.,.01,(_t-1.20e-03))*(1.-(1.-.13)*smstep(0.,.12,(_t-1.20e-03)-.01))+3.72e-01*lpnoise((_t-2.40e-03),10000.)*smstep(0.,.01,(_t-2.40e-03))*(1.-(1.-.13)*smstep(0.,.12,(_t-2.40e-03)-.01))))*exp(-4.*max(_t-.25,0.));
                        amaydrumR = vel*vel*1.3*exp(-7.*max(_t2-.05+10.*vel,0.))*metalnoise(.6*_t2, .5, 2.)
      +vel*1.3*(lpnoise(_t,10000.)*smstep(0.,.01,_t)*(1.-(1.-.13)*smstep(0.,.12,_t-.01))-.3*(1.00*lpnoise((_t-0.00),10000.)*smstep(0.,.01,(_t-0.00))*(1.-(1.-.13)*smstep(0.,.12,(_t-0.00)-.01))+6.10e-01*lpnoise((_t-1.20e-03),10000.)*smstep(0.,.01,(_t-1.20e-03))*(1.-(1.-.13)*smstep(0.,.12,(_t-1.20e-03)-.01))+3.72e-01*lpnoise((_t-2.40e-03),10000.)*smstep(0.,.01,(_t-2.40e-03))*(1.-(1.-.13)*smstep(0.,.12,(_t-2.40e-03)-.01))))*exp(-4.*max(_t2-.25,0.));
                    }
                    else if(drum == 5){
                        amaydrumL = vel*.9*protokick(_t,242.,55.,.036,.03,.0666,1.42,.02,.25,.01,.1,.4)
      +.9*protokick(_t,3333.,340.,.008,0.,.01,2.,2.4,0.,.2,.3,1.)
      +.64*((clamp(2.27*_tri(drop_phase(_t,.03,241.,72.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t-.01))+.91*clamp(.9*_tri(drop_phase(_t,.03,241.,72.)+.91*lpnoise(_t,8164.)),-1.,1.)*exp(-20.76*_t)+.05*lpnoise(_t,10466.)*(1.-smstep(0.,.18,_t-.56))+.56*lpnoise(_t,7123.)*exp(-_t*5.45)+.11*lpnoise(_t,1134.)*exp(-_t*13.82))*smstep(0.,.004,_t));
                        amaydrumR = vel*.9*protokick(_t2,242.,55.,.036,.03,.0666,1.42,.02,.25,.01,.1,.4)
      +.9*protokick(_t2,3333.,340.,.008,0.,.01,2.,2.4,0.,.2,.3,1.)
      +.64*((clamp(2.27*_tri(drop_phase(_t2,.03,241.,72.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t2-.01))+.91*clamp(.9*_tri(drop_phase(_t2,.03,241.,72.)+.91*lpnoise(_t2,8164.)),-1.,1.)*exp(-20.76*_t2)+.05*lpnoise(_t2,10466.)*(1.-smstep(0.,.18,_t2-.56))+.56*lpnoise(_t2,7123.)*exp(-_t2*5.45)+.11*lpnoise(_t2,1134.)*exp(-_t2*13.82))*smstep(0.,.004,_t2));
                    }
                    else if(drum == 6){
                        amaydrumL = vel*(vel*(_HCybHHClENV0(_t)*(.13*sinshape(pseudorandom(_HCybHHClENV2(_t)*_t+1.*_HCybHHClENV0(_t)*(.5*lpnoise(_t,14142.828)+.5*lpnoise(_t,.463*14142.828))),_HCybHHClENV3(_t),3.))+_HCybHHClENV1(_t)*(.04*(.5*lpnoise(_t,14142.828)+.5*lpnoise(_t,.463*14142.828)))));
                        amaydrumR = vel*(vel*(_HCybHHClENV0(_t)*(.13*sinshape(pseudorandom(_HCybHHClENV2(_t)*(_t-.00044)+1.*_HCybHHClENV0(_t)*(.5*lpnoise((_t-.00044),14142.828)+.5*lpnoise((_t-.00044),.463*14142.828))),_HCybHHClENV3(_t),3.))+_HCybHHClENV1(_t)*(.04*(.5*lpnoise((_t-.00127),14142.828)+.5*lpnoise((_t-.00127),.463*14142.828)))));
                    }
                    else if(drum == 7){
                        amaydrumL = vel*.85*(clamp(1.15*_tri(drop_phase(_t,.13,157.,76.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t-.13))+.81*clamp(.24*_tri(drop_phase(_t,.13,157.,76.)+.81*lpnoise(_t,2401.)),-1.,1.)*exp(-14.8*_t)+.01*lpnoise(_t,4079.)*(1.-smstep(0.,.7,_t-.12))+.5*lpnoise(_t,5164.)*exp(-_t*19.79)+.76*lpnoise(_t,8446.)*exp(-_t*24.))*smstep(0.,.002,_t);
                        amaydrumR = vel*.85*(clamp(1.15*_tri(drop_phase(_t2,.13,157.,76.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t2-.13))+.81*clamp(.24*_tri(drop_phase(_t2,.13,157.,76.)+.81*lpnoise(_t2,2401.)),-1.,1.)*exp(-14.8*_t2)+.01*lpnoise(_t2,4079.)*(1.-smstep(0.,.7,_t2-.12))+.5*lpnoise(_t2,5164.)*exp(-_t2*19.79)+.76*lpnoise(_t2,8446.)*exp(-_t2*24.))*smstep(0.,.002,_t2);
                    }
                    else if(drum == 37){
                        amaydrumL = vel*((clamp(1.35*_tri(drop_phase(_t,.07,244.,112.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t-.09))+.27*clamp(.03*_tri(drop_phase(_t,.07,244.,112.)+.27*lpnoise(_t,5148.)),-1.,1.)*exp(-24.07*_t)+0.*lpnoise(_t,1959.)*(1.-smstep(0.,.98,_t-.64))+.43*lpnoise(_t,8238.)*exp(-_t*25.54)+.11*lpnoise(_t,3803.)*exp(-_t*10.51))*smstep(0.,.006,_t))
      +((clamp(1.72*_tri(drop_phase(_t,.1,176.,66.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t-.13))+.06*clamp(.83*_tri(drop_phase(_t,.1,176.,66.)+.06*lpnoise(_t,6942.)),-1.,1.)*exp(-24.14*_t)+.07*lpnoise(_t,5757.)*(1.-smstep(0.,.84,_t-.18))+.6*lpnoise(_t,8812.)*exp(-_t*6.46)+.57*lpnoise(_t,4023.)*exp(-_t*9.28))*smstep(0.,.004,_t));
                        amaydrumR = vel*((clamp(1.35*_tri(drop_phase(_t2,.07,244.,112.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t2-.09))+.27*clamp(.03*_tri(drop_phase(_t2,.07,244.,112.)+.27*lpnoise(_t2,5148.)),-1.,1.)*exp(-24.07*_t2)+0.*lpnoise(_t2,1959.)*(1.-smstep(0.,.98,_t2-.64))+.43*lpnoise(_t2,8238.)*exp(-_t2*25.54)+.11*lpnoise(_t2,3803.)*exp(-_t2*10.51))*smstep(0.,.006,_t2))
      +((clamp(1.72*_tri(drop_phase(_t2,.1,176.,66.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t2-.13))+.06*clamp(.83*_tri(drop_phase(_t2,.1,176.,66.)+.06*lpnoise(_t2,6942.)),-1.,1.)*exp(-24.14*_t2)+.07*lpnoise(_t2,5757.)*(1.-smstep(0.,.84,_t2-.18))+.6*lpnoise(_t2,8812.)*exp(-_t2*6.46)+.57*lpnoise(_t2,4023.)*exp(-_t2*9.28))*smstep(0.,.004,_t2));
                    }
                    else if(drum == 38){
                        amaydrumL = vel*(theta(Bprog)*exp(-3.*Bprog)*_tri(9301.*_t+env_AHDSR(Bprog,L,.21,.18,.54,.2,.3)*.1*pseudorandom(1.*_t)+10.*(2.*fract(3920.*_t+.4*pseudorandom(1.*_t))-1.)*env_AHDSR(Bprog,L,.3,.1,.6,0.,0.)));
                        amaydrumR = vel*(theta(Bprog)*exp(-3.*Bprog)*_tri(9301.*_t2+env_AHDSR(Bprog,L,.21,.18,.54,.2,.3)*.1*pseudorandom(1.*_t2)+10.*(2.*fract(3920.*_t2+.4*pseudorandom(1.*_t2))-1.)*env_AHDSR(Bprog,L,.3,.1,.6,0.,0.)));
                    }
                    else if(drum == 40){
                        amaydrumL = vel*((clamp(2.58*_tri(drop_phase(_t,.05,156.,91.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t-.04))+.35*clamp(.44*_tri(drop_phase(_t,.05,156.,91.)+.35*lpnoise(_t,6183.)),-1.,1.)*exp(-7.56*_t)+.08*lpnoise(_t,7245.)*(1.-smstep(0.,.58,_t-.2))+.72*lpnoise(_t,2591.)*exp(-_t*5.48)+.73*lpnoise(_t,3597.)*exp(-_t*29.4))*smstep(0.,.01,_t));
                        amaydrumR = vel*((clamp(2.58*_tri(drop_phase(_t2,.05,156.,91.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t2-.04))+.35*clamp(.44*_tri(drop_phase(_t2,.05,156.,91.)+.35*lpnoise(_t2,6183.)),-1.,1.)*exp(-7.56*_t2)+.08*lpnoise(_t2,7245.)*(1.-smstep(0.,.58,_t2-.2))+.72*lpnoise(_t2,2591.)*exp(-_t2*5.48)+.73*lpnoise(_t2,3597.)*exp(-_t2*29.4))*smstep(0.,.01,_t2));
                    }
                    else if(drum == 42){
                        amaydrumL = vel*(((clamp(1.42*_tri(drop_phase(_t,.02,261.,53.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t-.28))+.86*clamp(.04*_tri(drop_phase(_t,.02,261.,53.)+.86*lpnoise(_t,3210.)),-1.,1.)*exp(-20.77*_t)+.03*lpnoise(_t,2325.)*(1.-smstep(0.,.42,_t-.01))+.28*lpnoise(_t,4276.)*exp(-_t*14.57)+.72*lpnoise(_t,3219.)*exp(-_t*9.93))*smstep(0.,.007,_t)));
                        amaydrumR = vel*(((clamp(1.42*_tri(drop_phase(_t2,.02,261.,53.)),-1.,1.)*(1.-smstep(-1e-3,0.,_t2-.28))+.86*clamp(.04*_tri(drop_phase(_t2,.02,261.,53.)+.86*lpnoise(_t2,3210.)),-1.,1.)*exp(-20.77*_t2)+.03*lpnoise(_t2,2325.)*(1.-smstep(0.,.42,_t2-.01))+.28*lpnoise(_t2,4276.)*exp(-_t2*14.57)+.72*lpnoise(_t2,3219.)*exp(-_t2*9.93))*smstep(0.,.007,_t2)));
                    }
                    else if(drum == 44){
                        amaydrumL = vel*((.837*(.541*lpnoise(_t,2041.774)+.798*lpnoise(_t,8260.482)+.931*lpnoise(_t,8317.984))*(smstep(0.,.007,_t)-smstep(0.,.37,_t-.05))+_sin(drop_phase(_t,.033,464.443,270.029))*exp(-_t*32.249)*.841+_sin(drop_phase(_t*659.983,.033,464.443,270.029))*exp(-_t*33.)*.618));
                        amaydrumR = vel*((.837*(.541*lpnoise(_t,2041.774)+.798*lpnoise(_t,8260.482)+.931*lpnoise(_t,8317.984))*(smstep(0.,.007,_t)-smstep(0.,.37,_t-.05))+_sin(drop_phase(_t,.033,464.443,270.029))*exp(-_t*32.249)*.841+_sin(drop_phase(_t*659.983,.033,464.443,270.029))*exp(-_t*33.)*.618));
                    }
                    
                    if(drum > 0)
                    {
                        dL += amtL * s_atan(env * amaydrumL);
                        dR += amtR * s_atan(env * amaydrumR);
                    }
                }
                else
                {
                    f = freqC1(note_pitch(_note) + mod_transp(_mod));

                    if(abs(slide) > 1e-3) // THIS IS SLIDEY BIZ
                    {
                        float Bslide = trk_slide(trk);
                        float fac = slide * log(2.)/12.;
                        if (Bprog <= Bslide)
                        {
                            float help = 1. - Bprog/Bslide;
                            f *= Bslide * (fhelp(fac) - help * fhelp(fac*help*help)) / Bprog;
                        }
                        else
                        {
                            f *= 1. + (Bslide * (fhelp(fac)-1.)) / Bprog;
                        }
                    }

                    env = theta(Bprog) * (1. - smstep(Boff-rel, Boff, B));
                    if(syn == 0){amaysynL = _sin(f*_t); amaysynR = _sin(f*_t2);}
                    else if(syn == 6){
                        time2 = time-.02; _t2 = _t-.02;
                        amaysynL = (QFM(floor((11000-110.*aux)*(_t-0.0*(1.+3.*_sin(.1*_t)))+.5)/(11000-110.*aux),f,0.,.00787*53.,.00787*env_AHDSR(Bprog,L,.247*vel,.006,.165,.103,0.)*122.,.00787*env_AHDSR(Bprog,L,.041*vel,.149,.14,.278,0.)*57.,.00787*env_AHDSR(Bprog,L,.119*vel,.158,.245,.49,0.)*71.,.5,1.,1.001,1.,.00787*45.,.00787*91.,.00787*53.,.00787*123.,3.)
      +QFM(floor((11000-110.*aux)*(_t-4.0e-03*(1.+3.*_sin(.1*_t)))+.5)/(11000-110.*aux),f,0.,.00787*53.,.00787*env_AHDSR(Bprog,L,.247*vel,.006,.165,.103,0.)*122.,.00787*env_AHDSR(Bprog,L,.041*vel,.149,.14,.278,0.)*57.,.00787*env_AHDSR(Bprog,L,.119*vel,.158,.245,.49,0.)*71.,.5,1.,1.001,1.,.00787*45.,.00787*91.,.00787*53.,.00787*123.,3.)
      +QFM(floor((11000-110.*aux)*(_t-8.0e-03*(1.+3.*_sin(.1*_t)))+.5)/(11000-110.*aux),f,0.,.00787*53.,.00787*env_AHDSR(Bprog,L,.247*vel,.006,.165,.103,0.)*122.,.00787*env_AHDSR(Bprog,L,.041*vel,.149,.14,.278,0.)*57.,.00787*env_AHDSR(Bprog,L,.119*vel,.158,.245,.49,0.)*71.,.5,1.,1.001,1.,.00787*45.,.00787*91.,.00787*53.,.00787*123.,3.))*env_AHDSR(Bprog,L,.068*vel,.045,.098,.614,.035);
                        amaysynR = (QFM(floor((11000-110.*aux)*(_t2-0.0*(1.+3.*_sin(.1*_t2)))+.5)/(11000-110.*aux),f,0.,.00787*53.,.00787*env_AHDSR(Bprog,L,.247*vel,.006,.165,.103,0.)*122.,.00787*env_AHDSR(Bprog,L,.041*vel,.149,.14,.278,0.)*57.,.00787*env_AHDSR(Bprog,L,.119*vel,.158,.245,.49,0.)*71.,.5,1.,1.001,1.,.00787*45.,.00787*91.,.00787*53.,.00787*123.,3.)
      +QFM(floor((11000-110.*aux)*(_t2-4.0e-03*(1.+3.*_sin(.1*_t2)))+.5)/(11000-110.*aux),f,0.,.00787*53.,.00787*env_AHDSR(Bprog,L,.247*vel,.006,.165,.103,0.)*122.,.00787*env_AHDSR(Bprog,L,.041*vel,.149,.14,.278,0.)*57.,.00787*env_AHDSR(Bprog,L,.119*vel,.158,.245,.49,0.)*71.,.5,1.,1.001,1.,.00787*45.,.00787*91.,.00787*53.,.00787*123.,3.)
      +QFM(floor((11000-110.*aux)*(_t2-8.0e-03*(1.+3.*_sin(.1*_t2)))+.5)/(11000-110.*aux),f,0.,.00787*53.,.00787*env_AHDSR(Bprog,L,.247*vel,.006,.165,.103,0.)*122.,.00787*env_AHDSR(Bprog,L,.041*vel,.149,.14,.278,0.)*57.,.00787*env_AHDSR(Bprog,L,.119*vel,.158,.245,.49,0.)*71.,.5,1.,1.001,1.,.00787*45.,.00787*91.,.00787*53.,.00787*123.,3.))*env_AHDSR(Bprog,L,.068*vel,.045,.098,.614,.035);
                    }
                    else if(syn == 14){
                        
                        amaysynL = (1.0*env_limit_length((Bprog-0.000),.34*(L-rel),.05)*env_AHDSR((Bprog-0.000),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t-SPB*0.000),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +4.1e-01*env_limit_length((Bprog-6.300e-02),.34*(L-rel),.05)*env_AHDSR((Bprog-6.300e-02),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t-SPB*6.300e-02),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +1.7e-01*env_limit_length((Bprog-1.260e-01),.34*(L-rel),.05)*env_AHDSR((Bprog-1.260e-01),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t-SPB*1.260e-01),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +6.9e-02*env_limit_length((Bprog-1.890e-01),.34*(L-rel),.05)*env_AHDSR((Bprog-1.890e-01),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t-SPB*1.890e-01),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +2.9e-02*env_limit_length((Bprog-2.520e-01),.34*(L-rel),.05)*env_AHDSR((Bprog-2.520e-01),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t-SPB*2.520e-01),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.));
                        amaysynR = (1.0*env_limit_length((Bprog-0.000),.34*(L-rel),.05)*env_AHDSR((Bprog-0.000),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t2-SPB*0.000),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +4.1e-01*env_limit_length((Bprog-6.300e-02),.34*(L-rel),.05)*env_AHDSR((Bprog-6.300e-02),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t2-SPB*6.300e-02),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +1.7e-01*env_limit_length((Bprog-1.260e-01),.34*(L-rel),.05)*env_AHDSR((Bprog-1.260e-01),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t2-SPB*1.260e-01),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +6.9e-02*env_limit_length((Bprog-1.890e-01),.34*(L-rel),.05)*env_AHDSR((Bprog-1.890e-01),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t2-SPB*1.890e-01),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.)
      +2.9e-02*env_limit_length((Bprog-2.520e-01),.34*(L-rel),.05)*env_AHDSR((Bprog-2.520e-01),L,.002,0.,.1,.25,.13)*bandpassBPsaw01((_t2-SPB*2.520e-01),f,tL,vel,(2570.+621.+(621.*_sin(.44*BT))),87.,252.));
                    }
                    else if(syn == 23){
                        
                        amaysynL = env_AHDSR(Bprog,L,.2,0.,.1,1.,.9)*(1.0*(MADD((_t-SPB*0.000),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t-SPB*0.000),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*0.000),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*0.000),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*0.000),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*0.000),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)))
      +3.8e-01*(MADD((_t-SPB*6.400e-02),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t-SPB*6.400e-02),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*6.400e-02),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*6.400e-02),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*6.400e-02),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*6.400e-02),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)))
      +1.4e-01*(MADD((_t-SPB*1.280e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t-SPB*1.280e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.280e-01),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.280e-01),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.280e-01),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.280e-01),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)))
      +5.5e-02*(MADD((_t-SPB*1.920e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t-SPB*1.920e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.920e-01),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.920e-01),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.920e-01),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t-SPB*1.920e-01),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0))));
                        amaysynR = env_AHDSR(Bprog,L,.2,0.,.1,1.,.9)*(1.0*(MADD((_t2-SPB*0.000),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t2-SPB*0.000),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*0.000),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*0.000),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*0.000),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*0.000),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-0.000)))+300.*(.5+(.5*_sin(.33*(Bprog-0.000)))),100.,5.,100.,.001,0.,0.,0)))
      +3.8e-01*(MADD((_t2-SPB*6.400e-02),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t2-SPB*6.400e-02),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*6.400e-02),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*6.400e-02),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*6.400e-02),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*6.400e-02),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-6.400e-02)))+300.*(.5+(.5*_sin(.33*(Bprog-6.400e-02)))),100.,5.,100.,.001,0.,0.,0)))
      +1.4e-01*(MADD((_t2-SPB*1.280e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t2-SPB*1.280e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.280e-01),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.280e-01),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.280e-01),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.280e-01),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.280e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.280e-01)))),100.,5.,100.,.001,0.,0.,0)))
      +5.5e-02*(MADD((_t2-SPB*1.920e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)*.3+.5*s_atan(MADD((_t2-SPB*1.920e-01),f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.920e-01),1.01*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.920e-01),2.005*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.920e-01),4.02*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0)+MADD((_t2-SPB*1.920e-01),.49*f,0.,32,1,(-1.+(.6*Bproc)),(100.+(600.*(Bprog-1.920e-01)))+300.*(.5+(.5*_sin(.33*(Bprog-1.920e-01)))),100.,5.,100.,.001,0.,0.,0))));
                    }
                    else if(syn == 42){
                        
                        amaysynL = (env_AHDSR(Bprog,L,0.,.2,.2,0.,0.)*env_AHDSR(Bprog,L,.001,0.,.1,1.,0.)*(1.0*env_limit_length((Bprog-0.000),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t-SPB*0.000),.2*(2.*fract(2.*f*(_t-SPB*0.000)+.4*_tri(.5*f*(_t-SPB*0.000)))-1.))+_sq_(1.04*.25*f*(_t-SPB*0.000),.2*(2.*fract(2.*f*(_t-SPB*0.000)+.4*_tri(.5*f*(_t-SPB*0.000)))-1.)))+.8*(2.*fract(2.*f*(_t-SPB*0.000)+.4*_tri(.5*f*(_t-SPB*0.000)))-1.)),1e-05,.15,.13,.3,.8,.8)+3.0e-01*env_limit_length((Bprog-2.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t-SPB*2.000e-01),.2*(2.*fract(2.*f*(_t-SPB*2.000e-01)+.4*_tri(.5*f*(_t-SPB*2.000e-01)))-1.))+_sq_(1.04*.25*f*(_t-SPB*2.000e-01),.2*(2.*fract(2.*f*(_t-SPB*2.000e-01)+.4*_tri(.5*f*(_t-SPB*2.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t-SPB*2.000e-01)+.4*_tri(.5*f*(_t-SPB*2.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)+9.0e-02*env_limit_length((Bprog-4.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t-SPB*4.000e-01),.2*(2.*fract(2.*f*(_t-SPB*4.000e-01)+.4*_tri(.5*f*(_t-SPB*4.000e-01)))-1.))+_sq_(1.04*.25*f*(_t-SPB*4.000e-01),.2*(2.*fract(2.*f*(_t-SPB*4.000e-01)+.4*_tri(.5*f*(_t-SPB*4.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t-SPB*4.000e-01)+.4*_tri(.5*f*(_t-SPB*4.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)+2.7e-02*env_limit_length((Bprog-6.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t-SPB*6.000e-01),.2*(2.*fract(2.*f*(_t-SPB*6.000e-01)+.4*_tri(.5*f*(_t-SPB*6.000e-01)))-1.))+_sq_(1.04*.25*f*(_t-SPB*6.000e-01),.2*(2.*fract(2.*f*(_t-SPB*6.000e-01)+.4*_tri(.5*f*(_t-SPB*6.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t-SPB*6.000e-01)+.4*_tri(.5*f*(_t-SPB*6.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)+8.1e-03*env_limit_length((Bprog-8.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t-SPB*8.000e-01),.2*(2.*fract(2.*f*(_t-SPB*8.000e-01)+.4*_tri(.5*f*(_t-SPB*8.000e-01)))-1.))+_sq_(1.04*.25*f*(_t-SPB*8.000e-01),.2*(2.*fract(2.*f*(_t-SPB*8.000e-01)+.4*_tri(.5*f*(_t-SPB*8.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t-SPB*8.000e-01)+.4*_tri(.5*f*(_t-SPB*8.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)));
                        amaysynR = (env_AHDSR(Bprog,L,0.,.2,.2,0.,0.)*env_AHDSR(Bprog,L,.001,0.,.1,1.,0.)*(1.0*env_limit_length((Bprog-0.000),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t2-SPB*0.000),.2*(2.*fract(2.*f*(_t2-SPB*0.000)+.4*_tri(.5*f*(_t2-SPB*0.000)))-1.))+_sq_(1.04*.25*f*(_t2-SPB*0.000),.2*(2.*fract(2.*f*(_t2-SPB*0.000)+.4*_tri(.5*f*(_t2-SPB*0.000)))-1.)))+.8*(2.*fract(2.*f*(_t2-SPB*0.000)+.4*_tri(.5*f*(_t2-SPB*0.000)))-1.)),1e-05,.15,.13,.3,.8,.8)+3.0e-01*env_limit_length((Bprog-2.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t2-SPB*2.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*2.000e-01)+.4*_tri(.5*f*(_t2-SPB*2.000e-01)))-1.))+_sq_(1.04*.25*f*(_t2-SPB*2.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*2.000e-01)+.4*_tri(.5*f*(_t2-SPB*2.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t2-SPB*2.000e-01)+.4*_tri(.5*f*(_t2-SPB*2.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)+9.0e-02*env_limit_length((Bprog-4.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t2-SPB*4.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*4.000e-01)+.4*_tri(.5*f*(_t2-SPB*4.000e-01)))-1.))+_sq_(1.04*.25*f*(_t2-SPB*4.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*4.000e-01)+.4*_tri(.5*f*(_t2-SPB*4.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t2-SPB*4.000e-01)+.4*_tri(.5*f*(_t2-SPB*4.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)+2.7e-02*env_limit_length((Bprog-6.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t2-SPB*6.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*6.000e-01)+.4*_tri(.5*f*(_t2-SPB*6.000e-01)))-1.))+_sq_(1.04*.25*f*(_t2-SPB*6.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*6.000e-01)+.4*_tri(.5*f*(_t2-SPB*6.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t2-SPB*6.000e-01)+.4*_tri(.5*f*(_t2-SPB*6.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)+8.1e-03*env_limit_length((Bprog-8.000e-01),1.*(L-rel),.07)*waveshape((s_atan(_sq_(.25*f*(_t2-SPB*8.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*8.000e-01)+.4*_tri(.5*f*(_t2-SPB*8.000e-01)))-1.))+_sq_(1.04*.25*f*(_t2-SPB*8.000e-01),.2*(2.*fract(2.*f*(_t2-SPB*8.000e-01)+.4*_tri(.5*f*(_t2-SPB*8.000e-01)))-1.)))+.8*(2.*fract(2.*f*(_t2-SPB*8.000e-01)+.4*_tri(.5*f*(_t2-SPB*8.000e-01)))-1.)),1e-05,.15,.13,.3,.8,.8)));
                    }
                    else if(syn == 92){
                        time2 = time-1.3; _t2 = _t-1.3;
                        amaysynL = (1.0*((vel*QFM(((_t-SPB*0.000)-0.0*(1.+2.1*_sin(.8*(_t-SPB*0.000)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-0.000)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-0.000),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t-SPB*0.000)-2.0e-04*(1.+2.1*_sin(.8*(_t-SPB*0.000)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-0.000)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-0.000),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-0.000),.5*(L-rel),1.)
      +1.9e-01*((vel*QFM(((_t-SPB*2.430e-01)-0.0*(1.+2.1*_sin(.8*(_t-SPB*2.430e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-2.430e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-2.430e-01),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t-SPB*2.430e-01)-2.0e-04*(1.+2.1*_sin(.8*(_t-SPB*2.430e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-2.430e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-2.430e-01),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-2.430e-01),.5*(L-rel),1.)
      +3.6e-02*((vel*QFM(((_t-SPB*4.860e-01)-0.0*(1.+2.1*_sin(.8*(_t-SPB*4.860e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-4.860e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-4.860e-01),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t-SPB*4.860e-01)-2.0e-04*(1.+2.1*_sin(.8*(_t-SPB*4.860e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-4.860e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-4.860e-01),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-4.860e-01),.5*(L-rel),1.)
      +6.9e-03*((vel*QFM(((_t-SPB*7.290e-01)-0.0*(1.+2.1*_sin(.8*(_t-SPB*7.290e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-7.290e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-7.290e-01),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t-SPB*7.290e-01)-2.0e-04*(1.+2.1*_sin(.8*(_t-SPB*7.290e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-7.290e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-7.290e-01),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-7.290e-01),.5*(L-rel),1.));
                        amaysynR = (1.0*((vel*QFM(((_t2-SPB*0.000)-0.0*(1.+2.1*_sin(.8*(_t2-SPB*0.000)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-0.000)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-0.000),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t2-SPB*0.000)-2.0e-04*(1.+2.1*_sin(.8*(_t2-SPB*0.000)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-0.000),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-0.000)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-0.000),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-0.000),.5*(L-rel),1.)
      +1.9e-01*((vel*QFM(((_t2-SPB*2.430e-01)-0.0*(1.+2.1*_sin(.8*(_t2-SPB*2.430e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-2.430e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-2.430e-01),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t2-SPB*2.430e-01)-2.0e-04*(1.+2.1*_sin(.8*(_t2-SPB*2.430e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-2.430e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-2.430e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-2.430e-01),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-2.430e-01),.5*(L-rel),1.)
      +3.6e-02*((vel*QFM(((_t2-SPB*4.860e-01)-0.0*(1.+2.1*_sin(.8*(_t2-SPB*4.860e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-4.860e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-4.860e-01),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t2-SPB*4.860e-01)-2.0e-04*(1.+2.1*_sin(.8*(_t2-SPB*4.860e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-4.860e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-4.860e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-4.860e-01),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-4.860e-01),.5*(L-rel),1.)
      +6.9e-03*((vel*QFM(((_t2-SPB*7.290e-01)-0.0*(1.+2.1*_sin(.8*(_t2-SPB*7.290e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-7.290e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-7.290e-01),L,.0001,.044,.163,.13,.129))
      +(vel*QFM(((_t2-SPB*7.290e-01)-2.0e-04*(1.+2.1*_sin(.8*(_t2-SPB*7.290e-01)))),f,0.,.00787*85.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.07,.093,.148,0.)*33.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.285,.2,.357,0.)*68.,.00787*env_AHDSR((Bprog-7.290e-01),L,.0001,.244,.181,.003,0.)*120.,.999,1.,1.+.0849*(.5+(.5*_sin(.25*(Bprog-7.290e-01)))),2.,.00787*10.,.00787*53.,.00787*115.,.00787*38.,9.)*env_AHDSR((Bprog-7.290e-01),L,.0001,.044,.163,.13,.129)))*env_limit_length((Bprog-7.290e-01),.5*(L-rel),1.));
env = theta(Bprog)*pow(1.-smstep(Boff-rel, Boff, B),.3);
                    }
                    else if(syn == 108){
                        time2 = time-2e-2; _t2 = _t-2e-2;
                        amaysynL = .25*clamp(1.+(.13-Bprog)/(.01),exp(-7.*Bprog),1.)*clip((1.+5.+4.*(.5+(.5*_sin(.2*BT)))*(.5+(.5*_sin(.21*BT))))*_tri(.251*f*_t+.35*(.5+(.5*_sin(.5*Bprog)))))
      +.76*clamp(1.+(.13-Bprog)/(.01),exp(-7.*Bprog),1.)*_tri(.5*f*_t+.4+.15*(.5+(.5*_sin(.5*Bprog)))+.22*env_AHDSR(Bprog,L,.025,0.,.1,.3,0.)*clip((1.+5.+4.*(.5+(.5*_sin(.2*BT)))*(.5+(.5*_sin(.21*BT))))*_tri(.251*f*_t+.35*(.5+(.5*_sin(.5*Bprog))))));
                        amaysynR = .25*clamp(1.+(.13-Bprog)/(.01),exp(-7.*Bprog),1.)*clip((1.+5.+4.*(.5+(.5*_sin(.2*BT)))*(.5+(.5*_sin(.21*BT))))*_tri(.251*f*_t2+.35*(.5+(.5*_sin(.5*Bprog)))))
      +.76*clamp(1.+(.13-Bprog)/(.01),exp(-7.*Bprog),1.)*_tri(.5*f*_t2+.4+.15*(.5+(.5*_sin(.5*Bprog)))+.22*env_AHDSR(Bprog,L,.025,0.,.1,.3,0.)*clip((1.+5.+4.*(.5+(.5*_sin(.2*BT)))*(.5+(.5*_sin(.21*BT))))*_tri(.251*f*_t2+.35*(.5+(.5*_sin(.5*Bprog))))));
                    }
                    else if(syn == 118){
                        
                        amaysynL = (vel*env_AHDSR(Bprog,L,.025,0.,.124,.55,.013)*MADD(floor(16000.*_t+.5)/16000.,f,0.,128,1,.23,(1158.+(818.*_sin_(2.*BT,.4))),16.,0.,1.98,.023,.4*(.55+(.4*clip((1.+1.)*_sin(4.*BT)))),0.,0));
                        amaysynR = (vel*env_AHDSR(Bprog,L,.025,0.,.124,.55,.013)*MADD(floor(16000.*_t2+.5)/16000.,f,0.,128,1,.23,(1158.+(818.*_sin_(2.*BT,.4))),16.,0.,1.98,.023,.4*(.55+(.4*clip((1.+1.)*_sin(4.*BT)))),0.,0));
                    }
                    
                    sL += amtL * trk_norm(trk) * s_atan(clamp(env,0.,1.) * amaysynL);
                    sR += amtR * trk_norm(trk) * s_atan(clamp(env,0.,1.) * amaysynR);
                }
            }
        }
    }
    return .35 * sidechain * vec2(s_atan(sL), s_atan(sR)) + .88 * vec2(s_atan(dL), s_atan(dR));
}

void main()
{
    Tsample = 1./iSampleRate;
    float t = (iBlockOffset + gl_FragCoord.x + gl_FragCoord.y*iTexSize) * Tsample;
    vec2 s = mainSynth(t);
    vec2 v  = floor((0.5+0.5*s)*65535.0);
    vec2 vl = mod(v,256.0)/255.0;
    vec2 vh = floor(v/256.0)/255.0;
    gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y);
}
