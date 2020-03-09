/* cozy - 4k executable gfx entry by NR4/Team210, shown at Under Construction 2k19
 * Copyright (C) 2019  Alexander Kraus <nr4@z10.info>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// #define DEBUG

#include <stdio.h> 
#include <stdint.h>
#include <windows.h>
#include <commctrl.h>
#include "GL/GL.h"
#include "glext.h"


PFNGLCREATESHADERPROC glCreateShader;
PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLUNIFORM1FPROC glUniform1f;
PFNGLUNIFORM1IPROC glUniform1i;
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
PFNGLACTIVETEXTUREPROC glActiveTexture;
#ifdef DEBUG
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLGETPROGRAMIVPROC glGetProgramiv;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
#endif

#ifdef DEBUG
// TODO: remove below
void debug(int shader_handle)
{
	printf("    Debugging shader with handle %d.\n", shader_handle);
	int compile_status = 0;
	glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &compile_status);
	if(compile_status != GL_TRUE)
	{
		printf("    FAILED.\n");
		GLint len;
		glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &len);
		printf("    Log length: %d\n", len);
		GLchar *CompileLog = (GLchar*)malloc(len*sizeof(GLchar));
		glGetShaderInfoLog(shader_handle, len, NULL, CompileLog);
		printf("    Error messages:\n%s\n", CompileLog);
		free(CompileLog);
	}
	else
		printf("    Shader compilation successful.\n");
}

void debugp(int program)
{
	printf("    Debugging program with handle %d.\n", program);
	int compile_status = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &compile_status);
	if(compile_status != GL_TRUE)
	{
		printf("    FAILED.\n");
		GLint len;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
		printf("    Log length: %d\n", len);
		GLchar *CompileLog = (GLchar*)malloc(len*sizeof(GLchar));
		glGetProgramInfoLog(program, len, NULL, CompileLog);
		printf("    Error messages:\n%s\n", CompileLog);
		free(CompileLog);
	}
	else
		printf("    Program linking successful.\n");
}
#endif

#define sprintf wsprintfA

PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    //Flags
		PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
		32,                   // Colordepth of the framebuffer.
		0, 0, 0, 0, 0, 0,
		0,
		0,
		0,
		0, 0, 0, 0,
		24,                   // Number of bits for the depthbuffer
		8,                    // Number of bits for the stencilbuffer
		0,                    // Number of Aux buffers in the framebuffer.
		PFD_MAIN_PLANE,
		0,
		0, 0, 0
	};

const int bufferSizes[] = {64, 128, 256, 512, 1024},
    nBufferSizes = 5;

size_t strlen(const char *str)
{
	int len = 0;
	while(str[len] != '\0') ++len;
	return len;
}

void *memset(void *ptr, int value, size_t num)
{
	for(int i=num-1; i>=0; i--)
		((unsigned char *)ptr)[i] = value;
	return ptr;
}

void *malloc(size_t size)
{
	return GlobalAlloc(GMEM_ZEROINIT, size);
}

int generated = 0,
    texs = 512,
    block_size = 512 * 512,
    channels = 2,
    nblocks1,
    sequence_texture_handle,
    snd_framebuffer,
    snd_texture,
    sample_rate = 44100,
    music1_size,
    sfx_handle,
    sfx_program,
    sfx_samplerate_location,
    sfx_blockoffset_location,
    sfx_volumelocation,
    sfx_texs_location,
    sfx_sequence_texture_location,
    sfx_sequence_width_location,
    paused = 1;
float duration1 = 188.,
    *smusic1;
HWND hPrecalcLoadingBar,
    hPlayPauseButton,
    generateButton;
HWAVEOUT hWaveOut;
WAVEHDR header = { 0, 0, 0, 0, 0, 0, 0, 0 };
    
#include "sequence.h"
#include "sfx.h"
#define SFX_VAR_ISAMPLERATE "iSampleRate"
#define SFX_VAR_IBLOCKOFFSET "iBlockOffset"
#define SFX_VAR_IVOLUME "iVolume"
#define SFX_VAR_ITEXSIZE "iTexSize"
#define SFX_VAR_ISEQUENCE "iSequence"
#define SFX_VAR_ISEQUENCEWIDTH "iSequenceWidth"

LRESULT CALLBACK DialogProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    int selectedIndex = 3;
    HINSTANCE hInstance = GetWindowLong(hwnd, GWL_HINSTANCE);
    HDC hdc = GetDC(hwnd);
    
	switch(uMsg)
	{
		case WM_COMMAND:
			UINT id =  LOWORD(wParam);
			HWND hSender = (HWND)lParam;

			switch(id)
			{
                case 5: // SFX buffer size combo box
                    selectedIndex = SendMessage(hSender, CB_GETCURSEL, 0, 0);
                    texs = bufferSizes[selectedIndex];
                    block_size = texs * texs;
                    break;
                case 6: // Generate button
                    glGenTextures(1, &sequence_texture_handle);
                    glBindTexture(GL_TEXTURE_2D, sequence_texture_handle);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sequence_texture_size, sequence_texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, sequence_texture);
                    glGenFramebuffers(1, &snd_framebuffer);
                    glBindFramebuffer(GL_FRAMEBUFFER, snd_framebuffer);
                    glPixelStorei(GL_PACK_ALIGNMENT, 4);
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

                    glGenTextures(1, &snd_texture);
                    glBindTexture(GL_TEXTURE_2D, snd_texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texs, texs, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, snd_texture, 0);
                    
                    // Music allocs
                    nblocks1 = sample_rate * duration1 / block_size + 1;
                    music1_size = nblocks1 * block_size;
                    smusic1 = (float*)malloc(4 * music1_size);
                    short *dest = (short*)smusic1;
                    for (int i = 0; i < 2 * music1_size; ++i)
                        dest[i] = 0;

                    // Load music shader
                    int sfx_size = strlen(sfx_frag);
                    sfx_handle = glCreateShader(GL_FRAGMENT_SHADER);
                    glShaderSource(sfx_handle, 1, (GLchar **)&sfx_frag, &sfx_size);
                    glCompileShader(sfx_handle);
#ifdef DEBUG
                    debug(sfx_handle);
#endif
                    sfx_program = glCreateProgram();
                    glAttachShader(sfx_program, sfx_handle);
                    glLinkProgram(sfx_program);
#ifdef DEBUG
                    debugp(sfx_program);
#endif
                    glUseProgram(sfx_program);
                    sfx_samplerate_location = glGetUniformLocation(sfx_program, SFX_VAR_ISAMPLERATE);
                    sfx_blockoffset_location = glGetUniformLocation(sfx_program, SFX_VAR_IBLOCKOFFSET);
                    sfx_texs_location = glGetUniformLocation(sfx_program, SFX_VAR_ITEXSIZE);
                    sfx_sequence_texture_location = glGetUniformLocation(sfx_program, SFX_VAR_ISEQUENCE);
                    sfx_sequence_width_location = glGetUniformLocation(sfx_program, SFX_VAR_ISEQUENCEWIDTH);
                    
                    
                    SendMessage(hPrecalcLoadingBar, PBM_SETRANGE, 0, MAKELPARAM(0, nblocks1));
                    SendMessage(hPrecalcLoadingBar, PBM_SETSTEP, (WPARAM) 1, 0); 
                    
                    for (int music_block = 0; music_block < nblocks1; ++music_block)
                    {
                        glViewport(0, 0, texs, texs);
                        
                        double tstart = (double)(music_block*block_size);

                        glUniform1f(sfx_volumelocation, 1.);
                        glUniform1f(sfx_samplerate_location, (float)sample_rate);
                        glUniform1f(sfx_blockoffset_location, (float)tstart);
                        glUniform1f(sfx_texs_location, (float)texs);
                        glUniform1i(sfx_sequence_texture_location, 0);
                        glUniform1f(sfx_sequence_width_location, (float)sequence_texture_size);

                        glActiveTexture(GL_TEXTURE0);
                        glBindTexture(GL_TEXTURE_2D, sequence_texture_handle);
                        
                        glBegin(GL_QUADS);
                        glVertex3f(-1,-1,0);
                        glVertex3f(-1,1,0);
                        glVertex3f(1,1,0);
                        glVertex3f(1,-1,0);
                        glEnd();
                        
                        glReadPixels(0, 0, texs, texs, GL_RGBA, GL_UNSIGNED_BYTE, smusic1 + music_block * block_size);
                        glFlush();

                        unsigned short *buf = (unsigned short*)smusic1;
                        short *dest = (short*)smusic1;
                        for (int j = 2 * music_block*block_size; j < 2 * (music_block + 1)*block_size; ++j)
                            dest[j] = (buf[j] - (1 << 15));
                        
                        SendMessage(hPrecalcLoadingBar, PBM_STEPIT, 0, 0);
                        RedrawWindow(hwnd, NULL, NULL, RDW_INTERNALPAINT);
                    }
                    
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    
                    int n_bits_per_sample = 16;
                    
                    FILE *f = fopen("music.raw", "wb");
                    fwrite(smusic1, sizeof(short), 2*nblocks1*block_size, f);
                    fclose(f);
                    
                    f = fopen("music.wav", "wb");
                    fwrite("RIFF", 1, 4, f);
                    int data = 44 + 2*nblocks1*block_size;
                    fwrite(&data, sizeof(int32_t), 1, f); // file size
//                     // WAV info section
                    fwrite("WAVEfmt ", 1, 8, f); // This stuff needs to be present
                    data = 16;
                    fwrite(&data, sizeof(int32_t), 1, f); // Remaining chunk with info is 16 bytes
                    data = 0x0001;
                    fwrite(&data, sizeof(int16_t), 1, f); // PCM format identifier
                    data = 2;
                    fwrite(&data, sizeof(int16_t), 1, f); // number of channels
                    data = sample_rate;
                    fwrite(&data, sizeof(int32_t), 1, f); // sample rate
                    data = sample_rate*n_bits_per_sample*channels/8;
                    fwrite(&data, sizeof(int32_t), 1, f); // bytes per second
                    data = channels*(n_bits_per_sample+7)/8;
                    fwrite(&data, sizeof(int16_t), 1, f); // frame size
                    data = n_bits_per_sample;
                    fwrite(&data, sizeof(int16_t), 1, f); // bits per sample
                    // Data section
                    fwrite("data", 1, 4, f);
                    data = 2*nblocks1*block_size;
                    fwrite(&data, sizeof(int32_t), 1, f); // data size
                    fwrite(smusic1, sizeof(short), 2*nblocks1*block_size, f); // data
                    fclose(f);
                    
                    EnableWindow(hPlayPauseButton, TRUE);
                    EnableWindow(generateButton, FALSE);
                    
                    hWaveOut = 0;
                    
                    WAVEFORMATEX wfx = { WAVE_FORMAT_PCM, channels, sample_rate, sample_rate*channels*n_bits_per_sample / 8, channels*n_bits_per_sample / 8, n_bits_per_sample, 0 };
                    waveOutOpen(&hWaveOut, WAVE_MAPPER, &wfx, 0, 0, CALLBACK_NULL);
                    header.lpData = smusic1;
                    header.dwBufferLength = 4 * (music1_size);
                    waveOutPrepareHeader(hWaveOut, &header, sizeof(WAVEHDR));
                    waveOutWrite(hWaveOut, &header, sizeof(WAVEHDR));
                    waveOutPause(hWaveOut);
                    
                    break;
                    
                case 8:
                    if(paused)
                    {
                        waveOutRestart(hWaveOut);
                        SetWindowText(hPlayPauseButton, "Pause");
                    }
                    else
                    {
                        waveOutPause(hWaveOut);
                        SetWindowText(hPlayPauseButton, "Play");
                    }
                    paused = !paused;
                    break;
            }
            break;
            
		case WM_CLOSE:
			ExitProcess(0);
			break;
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI demo(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
#ifdef DEBUG
    AllocConsole();
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);
#endif
    
    // Display demo window
	CHAR WindowClass[]  = "Team210 Demo Window";

	WNDCLASSEX wc = { 0 };
	wc.cbSize = sizeof(wc);
	wc.style = CS_OWNDC | CS_VREDRAW | CS_HREDRAW;
	wc.lpfnWndProc = &DialogProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = WindowClass;
	wc.hIconSm = NULL;

	RegisterClassEx(&wc);
    
    HWND hwnd = CreateWindowEx(0, WindowClass, ":: Team210 :: GO - MAKE A DEMO ::", WS_OVERLAPPEDWINDOW, 200, 200, 341, 150, NULL, NULL, hInstance, 0);
    
    // Add "SFX Buffer size: " text
	HWND hSFXBufferSizeText = CreateWindow(WC_STATIC, "SFX buffer size: ", WS_VISIBLE | WS_CHILD | SS_LEFT, 10,13,150,100, hwnd, NULL, hInstance, NULL);
    
    // Add SFX Buffer size combo box
    HWND hSFXBufferSizeComboBox = CreateWindow(WC_COMBOBOX, TEXT(""), CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE, 120, 10, 195, nBufferSizes*25, hwnd, (HMENU)5, hInstance,
	 NULL);
    for(int i=0; i<nBufferSizes; ++i)
    {
        char name[1024];
        sprintf(name, "%d pixels", bufferSizes[i]);
        SendMessage(hSFXBufferSizeComboBox, (UINT) CB_ADDSTRING, (WPARAM) 0, (LPARAM) name);
    }
    SendMessage(hSFXBufferSizeComboBox, CB_SETCURSEL, 3, 0);
    
    // Add Load button
    generateButton = CreateWindow(WC_BUTTON,"Generate",WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,10,35,100,25,hwnd,(HMENU)6,hInstance,NULL);
    
    // Add precalc loading bar
    hPrecalcLoadingBar = CreateWindowEx(0, PROGRESS_CLASS, (LPTSTR) NULL, WS_CHILD | WS_VISIBLE, 120, 36, 196, 25, hwnd, (HMENU) 7, hInstance, NULL);
    
    // Add a player trackbar
    HWND hTrackbar = CreateWindowEx(0,TRACKBAR_CLASS,"Music Trackbar",WS_CHILD | WS_VISIBLE, 111, 63, 213, 40, hwnd, (HMENU) 8, hInstance,NULL); 
    EnableWindow(hTrackbar, generated);
    SendMessage(hTrackbar, TBM_SETRANGE, 
        (WPARAM) TRUE,
        (LPARAM) MAKELONG(0, duration1));
    
    // Add Play button
    hPlayPauseButton = CreateWindow(WC_BUTTON,"Play",WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,10,65,100,25,hwnd,(HMENU)8,hInstance,NULL);
    EnableWindow(hPlayPauseButton, generated);
    
    ShowWindow(hwnd, TRUE);
	UpdateWindow(hwnd);

	HDC hdc = GetDC(hwnd);

	int  pf = ChoosePixelFormat(hdc, &pfd);
	SetPixelFormat(hdc, pf, &pfd);

	HGLRC glrc = wglCreateContext(hdc);
	wglMakeCurrent(hdc, glrc);

    glCreateShader = (PFNGLCREATESHADERPROC) wglGetProcAddress("glCreateShader");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC) wglGetProcAddress("glCreateProgram");
    glShaderSource = (PFNGLSHADERSOURCEPROC) wglGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC) wglGetProcAddress("glCompileShader");
    glAttachShader = (PFNGLATTACHSHADERPROC) wglGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC) wglGetProcAddress("glLinkProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC) wglGetProcAddress("glUseProgram");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC) wglGetProcAddress("glGetUniformLocation");
    glUniform1f = (PFNGLUNIFORM1FPROC) wglGetProcAddress("glUniform1f");
    glUniform1i = (PFNGLUNIFORM1IPROC) wglGetProcAddress("glUniform1i");
    glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC) wglGetProcAddress("glGenFramebuffers");
    glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC) wglGetProcAddress("glBindFramebuffer");
    glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC) wglGetProcAddress("glFramebufferTexture2D");
    glActiveTexture = (PFNGLACTIVETEXTUREPROC) wglGetProcAddress("glActiveTexture");
#ifdef DEBUG
    glGetShaderiv = (PFNGLGETSHADERIVPROC) wglGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC) wglGetProcAddress("glGetShaderInfoLog");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC) wglGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC) wglGetProcAddress("glGetProgramInfoLog");
#endif
    
    // Set timer for continuous progress bar update
    SetTimer(hwnd, 1337, 60, (TIMERPROC) NULL);
    
	MSG msg = { 0 };
	while(GetMessage(&msg, NULL, 0, 0) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
        
        if(!paused)
        {
            static MMTIME MMTime = { TIME_SAMPLES, 0};
            waveOutGetPosition(hWaveOut, &MMTime, sizeof(MMTIME));
            double time = 0 + ((double)MMTime.u.sample) / sample_rate;
            
            SendMessage(hTrackbar, TBM_SETPOS, 
                (WPARAM) TRUE,                   // redraw flag 
                (LPARAM) time); 
        }
	}
    
    return 0;
}
