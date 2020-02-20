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

#include <stdio.h> // FIXME: remove!
#include <windows.h>
#include <commctrl.h>
#include "GL/GL.h"
#include "glext.h"

#define sprintf wsprintfA

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
    nblocks1,
    sequence_texture_handle,
    snd_framebuffer,
    snd_texture,
    sample_rate = 48000,
    music1_size,
    sfx_handle,
    sfx_program,
    sfx_samplerate_location,
    sfx_blockoffset_location,
    sfx_volumelocation,
    sfx_texs_location,
    sfx_sequence_texture_location,
    sfx_sequence_width_location;
float duration1 = 180.,
    *smusic1;
    
#include "sequence.h"
#include "sfx.h"
#define SFX_VAR_IBLOCKOFFSET "iBlockOffset"
#define SFX_VAR_ISAMPLERATE "iSampleRate"
#define SFX_VAR_ITEXSIZE "iTexSize"
#define SFX_VAR_ISEQUENCE "iSequence"
#define SFX_VAR_ISEQUENCEWIDTH "iSequenceWidth"
#define SFX_VAR_IVOLUME "iVolume"

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
                    const GLchar *fragment_pointer = sfx_source;
                    sfx_handle = glCreateShader(GL_FRAGMENT_SHADER);
                    glShaderSource(sfx_handle, 1, &fragment_pointer, NULL);
                    glCompileShader(sfx_handle);
                    sfx_program = glCreateProgram();
                    glAttachShader(sfx_program, sfx_handle);
                    glLinkProgram(sfx_program);
                    glUseProgram(sfx_program);
                    sfx_samplerate_location = glGetUniformLocation(sfx_program, SFX_VAR_ISAMPLERATE);
                    sfx_blockoffset_location = glGetUniformLocation(sfx_program, SFX_VAR_IBLOCKOFFSET);
                    sfx_volumelocation = glGetUniformLocation(sfx_program, SFX_VAR_IVOLUME);
                    sfx_texs_location = glGetUniformLocation(sfx_program, SFX_VAR_ITEXSIZE);
                    sfx_sequence_texture_location = glGetUniformLocation(sfx_program, SFX_VAR_ISEQUENCE);
                    sfx_sequence_width_location = glGetUniformLocation(sfx_program, SFX_VAR_ISEQUENCEWIDTH);
                    
                    glViewport(0, 0, texs, texs);
                    
                    for (int music_block = 0; music_block < nblocks1; ++music_block)
                    {
                        printf("Rendering SFX block %d/%d -> %le\n", music_block, nblocks1, .5*(float)music_block / (float)nblocks1);
                        double tstart = (double)(music_block*block_size);

                        glUniform1f(sfx_volumelocation, 1.);
                        glUniform1f(sfx_samplerate_location, (float)sample_rate);
                        glUniform1f(sfx_blockoffset_location, (float)tstart);
                        glUniform1i(sfx_texs_location, texs);

                        glBegin(GL_QUADS);
                        glVertex3f(-1,-1,0);
                        glVertex3f(-1,1,0);
                        glVertex3f(1,1,0);
                        glVertex3f(1,-1,0);
                        glEnd();
                        
                        SwapBuffers(hdc);

                        glReadPixels(0, 0, texs, texs, GL_RGBA, GL_UNSIGNED_BYTE, smusic1 + music_block * block_size);
                        glFlush();

                        unsigned short *buf = (unsigned short*)smusic1;
                        short *dest = (short*)smusic1;
                        for (int j = 2 * music_block*block_size; j < 2 * (music_block + 1)*block_size; ++j)
                            dest[j] = (buf[j] - (1 << 15));
                    }
                    
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    RedrawWindow(hwnd, NULL, NULL, RDW_INTERNALPAINT);
                    
                    FILE *f = fopen("music.raw", "wt");
                    fwrite(smusic1, sizeof(short), 2*nblocks1*block_size, f);
                    fclose(f);
                    
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
    AllocConsole();
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);
    
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
    HWND generateButton = CreateWindow(WC_BUTTON,"Generate",WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,10,35,100,25,hwnd,(HMENU)6,hInstance,NULL);
    
    // Add precalc loading bar
    HWND hPrecalcLoadingBar = CreateWindowEx(0, PROGRESS_CLASS, (LPTSTR) NULL, WS_CHILD | WS_VISIBLE, 120, 36, 196, 25, hwnd, (HMENU) 7, hInstance, NULL);
    
    // Add a player trackbar
    HWND hTrackbar = CreateWindowEx(0,TRACKBAR_CLASS,"Music Trackbar",WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_ENABLESELRANGE, 111, 63, 213, 40, hwnd, (HMENU) 8, hInstance,NULL); 
    EnableWindow(hTrackbar, generated);
    
    // Add Play button
    HWND hPlayPauseButton = CreateWindow(WC_BUTTON,"Play",WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,10,65,100,25,hwnd,(HMENU)6,hInstance,NULL);
    EnableWindow(hPlayPauseButton, generated);
    
    ShowWindow(hwnd, TRUE);
	UpdateWindow(hwnd);
    
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

	MSG msg = { 0 };
	while(GetMessage(&msg, NULL, 0, 0) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
    
    return 0;
}
