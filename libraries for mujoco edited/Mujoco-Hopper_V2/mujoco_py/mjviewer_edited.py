import ctypes
from ctypes import pointer, byref
import logging
from threading import Lock
import os

from . import mjcore, mjconstants, glfw
from .mjlib import mjlib
import numpy as np
import OpenGL.GL as gl

logger = logging.getLogger(__name__)

mjCAT_ALL = 7

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import glfw
import freetype
import glm
from PIL import Image
import math
import time

def _glfw_error_callback(e, d):
    logger.error('GLFW error: %s, desc: %s', e, d)

class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')
        
class MjViewer(object):

    def __init__(self, visible=True, init_width=500, init_height=500, go_fast=False):
        """
        Set go_fast=True to run at full speed instead of waiting for the 60 Hz monitor refresh
        init_width and init_height set window size. On Mac Retina displays, they are in nominal
        pixels but .render returns an array of device pixels, so the array will be twice as big
        as you expect.
        """
        self.visible = visible
        self.init_width = init_width
        self.init_height = init_height
        self.go_fast = not visible or go_fast

        self.last_render_time = 0
        self.objects = mjcore.MJVOBJECTS()
        self.cam = mjcore.MJVCAMERA()
        self.vopt = mjcore.MJVOPTION()
        self.ropt = mjcore.MJROPTION()
        self.con = mjcore.MJRCONTEXT()
        self.running = False
        self.speedtype = 1
        self.window = None
        self.window2 = None
        self.model = None
        self.gui_lock = Lock()

        # framebuffer objects
        self._fbo = None
        self._rbo = None

        self._last_button = 0
        self._last_click_time = 0
        self._button_left_pressed = False
        self._button_middle_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0

    def set_model(self, model, tex):
        glfw.make_context_current(self.window)
        self.model = model
        if model:
            self.data = model.data
        else:
            self.data = None
        if self.running:
            if model:
                mjlib.mjr_makeContext(model.ptr, byref(self.con), 150)
            else:
                mjlib.mjr_makeContext(None, byref(self.con), 150)
            self.render(tex)
        if model:
            self.autoscale()

    def autoscale(self):
        glfw.make_context_current(self.window)
        self.cam.lookat[0] = self.model.stat.center[0]
        self.cam.lookat[1] = self.model.stat.center[1]
        self.cam.lookat[2] = self.model.stat.center[2]
        self.cam.distance = 0.5 * self.model.stat.extent
        self.cam.camid = -1
        self.cam.trackbodyid = 1
        width, height = self.get_dimensions()
        mjlib.mjv_updateCameraPose(byref(self.cam), width*1.0/height)

    def get_rect(self):
        rect = mjcore.MJRRECT(0, 0, 0, 0)
        rect.width, rect.height = self.get_dimensions()
        return rect

    #OPENGL MODE WINDOW

    # fontfile = r'C:\source\resource\fonts\gnu-freefont_freesans\freesans.ttf'

    def _get_rendering_buffer(self, xpos, ypos, w, h, zfix=0.0):
        return np.asarray([
            xpos, ypos - h, 0, 0,
            xpos, ypos, 0, 1,
                  xpos + w, ypos, 1, 1,
            xpos, ypos - h, 0, 0,
                  xpos + w, ypos, 1, 1,
                  xpos + w, ypos - h, 1, 0
        ], np.float32)

    def initliaze(self):

        global VERTEXT_SHADER
        global FRAGMENT_SHADER
        global shaderProgram
        global Characters
        global VBO
        global VAO

        shaderProgram = None
        Characters = dict()
        VBO = None
        VAO = None


        VERTEX_SHADER = """
                #version 330 core
                layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
                out vec2 TexCoords;

                uniform mat4 projection;

                void main()
                {
                    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
                    TexCoords = vertex.zw;
                }
               """

        FRAGMENT_SHADER = """
                #version 330 core
                in vec2 TexCoords;
                out vec4 color;

                uniform sampler2D text;
                uniform vec3 textColor;

                void main()
                {    
                    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
                    color = vec4(textColor, 1.0) * sampled;
                }
                """

        # compiling shaders
        vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

        # creating shaderProgram
        shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
        glUseProgram(shaderProgram)

        # get projection
        # problem

        shader_projection = glGetUniformLocation(shaderProgram, "projection")
        projection = glm.ortho(0, 640, 640, 0)
        glUniformMatrix4fv(shader_projection, 1, GL_FALSE, glm.value_ptr(projection))

        # disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        face = freetype.Face(r'C:\Windows\Fonts\arial.ttf')
        face.set_char_size(48 * 64)

        # load first 128 characters of ASCII set
        for i in range(0, 128):
            face.load_char(chr(i))
            glyph = face.glyph

            # generate texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, glyph.bitmap.width, glyph.bitmap.rows, 0,
                         GL_RED, GL_UNSIGNED_BYTE, glyph.bitmap.buffer)



            # texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # now store character for later use
            Characters[chr(i)] = CharacterSlot(texture, glyph)

        glBindTexture(GL_TEXTURE_2D, 0)

        # configure VAO/VBO for texture quads
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def render_text(self, window, text, x, y, scale, color):
        global shaderProgram
        global Characters
        global VBO
        global VAO

        face = freetype.Face(r'C:\Windows\Fonts\arial.ttf')
        face.set_char_size(48 * 64)
        glUniform3f(glGetUniformLocation(shaderProgram, "textColor"),
                    color[0] / 255, color[1] / 255, color[2] / 255)

        glActiveTexture(GL_TEXTURE0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(VAO)
        for c in text:
            ch = Characters[c]
            w, h = ch.textureSize
            w = w * scale
            h = h * scale
            vertices = self._get_rendering_buffer(x, y, w, h)

            # render glyph texture over quad
            glBindTexture(GL_TEXTURE_2D, ch.texture)
            # update content of VBO memory
            glBindBuffer(GL_ARRAY_BUFFER, VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            # render quad
            glDrawArrays(GL_TRIANGLES, 0, 6)
            # now advance cursors for next glyph (note that advance is number of 1/64 pixels)
            x += (ch.advance >> 6) * scale

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

        glfw.swap_buffers(window)
        glfw.poll_events()
## END OPENGL MODE WINDOW FUNCTIONS
    def render(self, tex):
        if not self.data:
            return
        glfw.make_context_current(self.window)

        self.gui_lock.acquire()
        rect = self.get_rect()
        arr = (ctypes.c_double*3)(0, 0, 0)

        mjlib.mjv_makeGeoms(self.model.ptr, self.data.ptr, byref(self.objects), byref(self.vopt), mjCAT_ALL, 0, None, None, ctypes.cast(arr, ctypes.POINTER(ctypes.c_double)))
        mjlib.mjv_makeLights(self.model.ptr, self.data.ptr, byref(self.objects))

        mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))

        mjlib.mjv_updateCameraPose(byref(self.cam), rect.width*1.0/rect.height)

        mjlib.mjr_render(0, rect, byref(self.objects), byref(self.ropt), byref(self.cam.pose), byref(self.con))

        self.gui_lock.release()

        # OPENGL MODE WINDOW CALL

        glfw.make_context_current(self.window2)
        self.initliaze()
        glfw.poll_events()
        glClearColor(0, 0, 0, 0)
        #glViewport(0, 0, 100, 100)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.render_text(self.window2, 'Mode: '+tex, 20, 50, 1, (255, 100, 100))

    def get_dimensions(self):
        """
        returns a tuple (width, height)
        """
        if self.window:
            return glfw.get_framebuffer_size(self.window)
        return (self.init_width, self.init_height)

    def get_image(self):
        """
        returns a tuple (data, width, height), where:
        - data is a string with raw bytes representing the pixels in 3-channel RGB
          (i.e. every three bytes = 1 pixel)
        - width is the width of the image
        - height is the height of the image
        """
        glfw.make_context_current(self.window)
        width, height = self.get_dimensions()
        gl.glReadBuffer(gl.GL_BACK)
        data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        return (data, width, height)

    def _init_framebuffer_object(self):
        """
        returns a Framebuffer Object to support offscreen rendering.
        http://learnopengl.com/#!Advanced-OpenGL/Framebuffers
        """
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        rbo = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER,
            gl.GL_RGBA,
            self.init_width,
            self.init_height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, rbo)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        fbo_status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)

        if fbo_status != gl.GL_FRAMEBUFFER_COMPLETE:
            gl.glDeleteFramebuffers([fbo])
            glfw.terminate()
            raise Exception('Framebuffer failed status check: %s' % fbo_status)

        self._fbo = fbo
        self._rbo = rbo

    def start(self):
        logger.info('initializing glfw@%s', glfw.get_version())

        glfw.set_error_callback(_glfw_error_callback)

        if not glfw.init():
            raise Exception('glfw failed to initialize')

        window = None
        if self.visible:
            glfw.window_hint(glfw.SAMPLES, 4)
            glfw.window_hint(glfw.VISIBLE, 1);
        else:
            glfw.window_hint(glfw.VISIBLE, 0);

        # try stereo if refresh rate is at least 100Hz
        stereo_available = False

        _, _, refresh_rate = glfw.get_video_mode(glfw.get_primary_monitor())
        if refresh_rate >= 100:
            glfw.window_hint(glfw.STEREO, 1)
            window = glfw.create_window(
                self.init_width, self.init_height, "Simulate", None, None)
            window2 = glfw.create_window(640, 640, "Modes", None, None)
            if window or window2:
                stereo_available = True

        # no stereo: try mono
        if not window:
            glfw.window_hint(glfw.STEREO, 0)
            window = glfw.create_window(self.init_width, self.init_height, "Simulate", None, None)
            window2 = glfw.create_window(640, 640, "Modes", None, None)
        if not window:
            glfw.terminate()
            return

        self.running = True

        # Make the window's context current
        glfw.make_context_current(window)

        if self.go_fast:
            # Let's go faster than 60 Hz
            glfw.swap_interval(0)

        self._init_framebuffer_object()

        width, height = glfw.get_framebuffer_size(window)
        width1, height = glfw.get_window_size(window)
        self._scale = width * 1.0 / width1

        self.window = window
        self.window2 = window2

        mjlib.mjv_makeObjects(byref(self.objects), 1000)

        mjlib.mjv_defaultCamera(byref(self.cam))
        mjlib.mjv_defaultOption(byref(self.vopt))
        mjlib.mjr_defaultOption(byref(self.ropt))

        mjlib.mjr_defaultContext(byref(self.con))

        if self.model:
            mjlib.mjr_makeContext(self.model.ptr, byref(self.con), 150)
            self.autoscale()
        else:
            mjlib.mjr_makeContext(None, byref(self.con), 150)

        glfw.set_cursor_pos_callback(window, self.handle_mouse_move)
        glfw.set_mouse_button_callback(window, self.handle_mouse_button)
        glfw.set_scroll_callback(window, self.handle_scroll)

        glfw.set_cursor_pos_callback(window2, self.handle_mouse_move)
        glfw.set_mouse_button_callback(window2, self.handle_mouse_button)
        glfw.set_scroll_callback(window2, self.handle_scroll)

    def handle_mouse_move(self, window, xpos, ypos):
        # no buttons down: nothing to do
        if not self._button_left_pressed \
                and not self._button_middle_pressed \
                and not self._button_right_pressed:
            return

        # compute mouse displacement, save
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(self.window)

        # get shift key state
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
                or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        # determine action based on mouse button
        action = None
        if self._button_right_pressed:
            action = mjconstants.MOUSE_MOVE_H if mod_shift else mjconstants.MOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mjconstants.MOUSE_ROTATE_H if mod_shift else mjconstants.MOUSE_ROTATE_V
        else:
            action = mjconstants.MOUSE_ZOOM

        self.gui_lock.acquire()

        mjlib.mjv_moveCamera(action, dx, dy, byref(self.cam), width, height)

        self.gui_lock.release()


    def handle_mouse_button(self, window, button, act, mods):
        # update button state
        self._button_left_pressed = \
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_middle_pressed = \
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._button_right_pressed = \
                glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        if not self.model:
            return

        self.gui_lock.acquire()

        # save info
        if act == glfw.PRESS:
            self._last_button = button
            self._last_click_time = glfw.get_time()

        self.gui_lock.release()

    def handle_scroll(self, window, x_offset, y_offset):
        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(window)

        # scroll
        self.gui_lock.acquire()
        mjlib.mjv_moveCamera(mjconstants.MOUSE_ZOOM, 0, (-20*y_offset), byref(self.cam), width, height)
        self.gui_lock.release()

    def should_stop(self):
        return glfw.window_should_close(self.window)

    def loop_once(self, tex):
        glfw.make_context_current(self.window)
        self.render(tex)
        # Swap front and back buffers
        glfw.swap_buffers(self.window)
        # Poll for and process events
        glfw.poll_events()

    def finish(self):
        glfw.make_context_current(self.window)
        glfw.destroy_window(self.window)

        glfw.make_context_current(self.window2)
        glfw.destroy_window(self.window2)

        if gl.glIsFramebuffer(self._fbo):
            gl.glDeleteFramebuffers(int(self._fbo))
        if gl.glIsRenderbuffer(self._rbo):
            gl.glDeleteRenderbuffers(1, int(self._rbo))

        mjlib.mjr_freeContext(byref(self.con))
        mjlib.mjv_freeObjects(byref(self.objects))
        self.running = False
