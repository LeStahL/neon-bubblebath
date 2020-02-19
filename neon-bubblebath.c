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
// PFNGLUNIFORM1IPROC glUniform1i;
// PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
// PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;

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
    music1_size;
float duration1 = 180.,
    *smusic1;
    
HWND hwnd;

#include "sfx.h"

LRESULT CALLBACK DialogProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    int selectedIndex = 3;
    
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
                    // Load SFX here.
//                     printf("sequence texture width is: %d\n", sequence_texture_size); // TODO: remove
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

                    unsigned int snd_texture;
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
                    sfx_program = glCreateProgram();
                    glShaderSource(sfx_handle, 1, (GLchar **)&sfx_frag, &sfx_size);
                    glCompileShader(sfx_handle);
//                     printf("---> SFX shader:\n");
//                 #ifdef DEBUG
//                     debug(sfx_handle);
//                 #endif
                    glAttachShader(sfx_program, sfx_handle);
                    glLinkProgram(sfx_program);
//                     printf("---> SFX program:\n");
//                 #ifdef DEBUG
//                     debugp(sfx_program);
//                 #endif
                    glUseProgram(sfx_program);
                    sfx_samplerate_location = glGetUniformLocation(sfx_program, SFX_VAR_ISAMPLERATE);
                    sfx_blockoffset_location = glGetUniformLocation(sfx_program, SFX_VAR_IBLOCKOFFSET);
                    sfx_volumelocation = glGetUniformLocation(sfx_program, SFX_VAR_IVOLUME);
                    sfx_texs_location = glGetUniformLocation(sfx_program, SFX_VAR_ITEXSIZE);
                    sfx_sequence_texture_location = glGetUniformLocation(sfx_program, SFX_VAR_ISEQUENCE);
                    sfx_sequence_width_location = glGetUniformLocation(sfx_program, SFX_VAR_ISEQUENCEWIDTH);
//                     printf("++++ SFX shader created.\n");

//                     glBindFramebuffer(GL_FRAMEBUFFER, 0);
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
    // Initialize OpenGL 
    hwnd = GetDesktopWindow();
    
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
    
    // ### Show selector
    WNDCLASS wca = { 0 };
	
    wca.lpfnWndProc   = DialogProc;
	wca.hInstance     = hInstance;
	wca.lpszClassName = L"Settings";
	RegisterClass(&wca);
    
	HWND lwnd = CreateWindowEx(0, L"Settings", "Neon Bubblebath", WS_OVERLAPPEDWINDOW, 200, 200, 341, 150, NULL, NULL, hInstance, NULL);
    
    // Add "SFX Buffer size: " text
	HWND hSFXBufferSizeText = CreateWindow(WC_STATIC, "SFX buffer size: ", WS_VISIBLE | WS_CHILD | SS_LEFT, 10,13,150,100, lwnd, NULL, hInstance, NULL);
    
    // Add SFX Buffer size combo box
    HWND hSFXBufferSizeComboBox = CreateWindow(WC_COMBOBOX, TEXT(""), CBS_DROPDOWN | CBS_HASSTRINGS | WS_CHILD | WS_OVERLAPPED | WS_VISIBLE, 120, 10, 195, nBufferSizes*25, lwnd, (HMENU)5, hInstance,
	 NULL);
    for(int i=0; i<nBufferSizes; ++i)
    {
        char name[1024];
        sprintf(name, "%d pixels", bufferSizes[i]);
        SendMessage(hSFXBufferSizeComboBox, (UINT) CB_ADDSTRING, (WPARAM) 0, (LPARAM) name);
    }
    SendMessage(hSFXBufferSizeComboBox, CB_SETCURSEL, 3, 0);
    
    // Add Load button
    HWND generateButton = CreateWindow(WC_BUTTON,"Generate",WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,10,35,100,25,lwnd,(HMENU)6,hInstance,NULL);
    
    // Add precalc loading bar
    HWND hPrecalcLoadingBar = CreateWindowEx(0, PROGRESS_CLASS, (LPTSTR) NULL, WS_CHILD | WS_VISIBLE, 120, 36, 196, 25, lwnd, (HMENU) 7, hInstance, NULL);
    
    // Add a player trackbar
    HWND hTrackbar = CreateWindowEx(0,TRACKBAR_CLASS,"Music Trackbar",WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_ENABLESELRANGE, 111, 63, 213, 40, lwnd, (HMENU) 8, hInstance,NULL); 
    EnableWindow(hTrackbar, generated);
    
    // Add Play button
    HWND hPlayPauseButton = CreateWindow(WC_BUTTON,"Play",WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,10,65,100,25,lwnd,(HMENU)6,hInstance,NULL);
    EnableWindow(hPlayPauseButton, generated);
    
    ShowWindow(lwnd, TRUE);
	UpdateWindow(lwnd);

	MSG msg = { 0 };
	while(GetMessage(&msg, NULL, 0, 0) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
    
    return 0;
}
