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
    sfxBufferSize = 512;
    
HWND hwnd;

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
                    sfxBufferSize = bufferSizes[selectedIndex];
                    break;
                case 6: // Generate button
                    // Load SFX here.
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
