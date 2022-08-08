#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <iostream>
#include <thread>

#include "viewer_cuda.h"

typedef unsigned char uchar;

std::mutex mtx;

class Viewer {
  public:
    Viewer(
      const torch::Tensor image,
      const torch::Tensor poses,
      const torch::Tensor points,
      const torch::Tensor colors,
      const torch::Tensor intrinsics);

    void close() {
      running = false;
    };

    void join() {
      tViewer.join();
    };

    void update_image(torch::Tensor img) {
      mtx.lock();
      redraw = true;
      image = img.permute({1,2,0}).to(torch::kCPU);
      mtx.unlock();
    }

    // main visualization
    void run();

  private:
    bool running;
    std::thread tViewer;

    int w;
    int h;
    int ux;

    int nPoints, nFrames;
    const torch::Tensor counter;
    const torch::Tensor dirty;

    torch::Tensor image;
    torch::Tensor poses;
    torch::Tensor points;
    torch::Tensor colors;
    torch::Tensor intrinsics;

    bool redraw;
    bool showForeground;
    bool showBackground;

    torch::Tensor transformMatrix;

    double filterThresh;
    void drawPoints();
    void drawPoses();

    void initVBO();
    void destroyVBO();

    // OpenGL buffers (vertex buffer, color buffer)
    GLuint vbo, cbo;
    struct cudaGraphicsResource *xyz_res;
    struct cudaGraphicsResource *rgb_res;

};

Viewer::Viewer(
      const torch::Tensor image,
      const torch::Tensor poses, 
      const torch::Tensor points,
      const torch::Tensor colors,
      const torch::Tensor intrinsics)
  : image(image), poses(poses), points(points), colors(colors), intrinsics(intrinsics)
{
  running = true;
  redraw = true;
  nFrames = poses.size(0);
  nPoints = points.size(0);

  ux = 0;
  h = image.size(0);
  w = image.size(1);

  tViewer = std::thread(&Viewer::run, this);
};

void Viewer::drawPoints() {
  float *xyz_ptr;
  uchar *rgb_ptr;
  size_t xyz_bytes;
  size_t rgb_bytes; 

  unsigned int size_xyz = 3 * points.size(0) * sizeof(float);
  unsigned int size_rgb = 3 * points.size(0) * sizeof(uchar);

  cudaGraphicsResourceGetMappedPointer((void **) &xyz_ptr, &xyz_bytes, xyz_res);
  cudaGraphicsResourceGetMappedPointer((void **) &rgb_ptr, &rgb_bytes, rgb_res);

  float *xyz_data = points.data_ptr<float>();
  cudaMemcpy(xyz_ptr, xyz_data, xyz_bytes, cudaMemcpyDeviceToDevice);

  uchar *rgb_data = colors.data_ptr<uchar>();
  cudaMemcpy(rgb_ptr, rgb_data, rgb_bytes, cudaMemcpyDeviceToDevice);

  // bind color buffer
  glBindBuffer(GL_ARRAY_BUFFER, cbo);
  glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(3, GL_FLOAT, 0, 0);

  // bind vertex buffer
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, points.size(0));
  glDisableClientState(GL_VERTEX_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glDisableClientState(GL_COLOR_ARRAY);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

}


void Viewer::drawPoses() {

  float *tptr = transformMatrix.data_ptr<float>();

  const int NUM_POINTS = 8;
  const int NUM_LINES = 10;

  const float CAM_POINTS[NUM_POINTS][3] = {
    { 0,   0,   0},
    {-1,  -1, 1.5},
    { 1,  -1, 1.5},
    { 1,   1, 1.5},
    {-1,   1, 1.5},
    {-0.5, 1, 1.5},
    { 0.5, 1, 1.5},
    { 0, 1.2, 1.5}};

  const int CAM_LINES[NUM_LINES][2] = {
    {1,2}, {2,3}, {3,4}, {4,1}, {1,0}, {0,2}, {3,0}, {0,4}, {5,7}, {7,6}};

  const float SZ = 0.05;

  glColor3f(0,0.5,1);
  glLineWidth(1.5);

  for (int i=0; i<nFrames; i++) {

      if (i + 1 == nFrames)
        glColor3f(1,0,0);

      glPushMatrix();
      glMultMatrixf((GLfloat*) (tptr + 4*4*i));

      glBegin(GL_LINES);
      for (int j=0; j<NUM_LINES; j++) {
        const int u = CAM_LINES[j][0], v = CAM_LINES[j][1];
        glVertex3f(SZ*CAM_POINTS[u][0], SZ*CAM_POINTS[u][1], SZ*CAM_POINTS[u][2]);
        glVertex3f(SZ*CAM_POINTS[v][0], SZ*CAM_POINTS[v][1], SZ*CAM_POINTS[v][2]);
      }
      glEnd();

      glPopMatrix();
  }
}

void Viewer::initVBO() {
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  // initialize buffer object
  unsigned int size_xyz = 3 * points.size(0) * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size_xyz, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&xyz_res, vbo, cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &xyz_res, 0);

  glGenBuffers(1, &cbo);
  glBindBuffer(GL_ARRAY_BUFFER, cbo);

  // initialize buffer object
  unsigned int size_rgb = 3 * points.size(0) * sizeof(uchar);
  glBufferData(GL_ARRAY_BUFFER, size_rgb, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&rgb_res, cbo, cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsMapResources(1, &rgb_res, 0);
}

void Viewer::destroyVBO() {
  cudaGraphicsUnmapResources(1, &xyz_res, 0);
  cudaGraphicsUnregisterResource(xyz_res);
  glBindBuffer(1, vbo);
  glDeleteBuffers(1, &vbo);

  cudaGraphicsUnmapResources(1, &rgb_res, 0);
  cudaGraphicsUnregisterResource(rgb_res);
  glBindBuffer(1, cbo);
  glDeleteBuffers(1, &cbo);
}

void Viewer::run() {

  // initialize OpenGL buffers

  pangolin::CreateWindowAndBind("DPVO", 2*640, 2*480);

	const int UI_WIDTH = 180;
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState Visualization3D_camera(
		pangolin::ProjectionMatrix(w, h,400,400,w/2,h/2,0.1,500),
		pangolin::ModelViewLookAt(-0,-1,-1, 0,0,0, pangolin::AxisNegY));

  pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w/(float)h)
		.SetHandler(new pangolin::Handler3D(Visualization3D_camera));

  initVBO();


  // UI Knobs
  // pangolin::CreatePanel("ui").SetBounds(0.8, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
  // pangolin::Var<double> settings_filterThresh("ui.filter",0.5,1e-2,1e2,true);
  // pangolin::Var<int> settings_frameIndex("ui.index", 0, 0, nFrames-1);
  // pangolin::Var<bool> settings_showSparse("ui.sparse",true,true);
  // pangolin::Var<bool> settings_showDense("ui.dense",true,true);
  // pangolin::Var<bool> settings_showForeground("ui.foreground",true,true);
  // pangolin::Var<bool> settings_showBackground("ui.background",true,true);


	pangolin::View& d_video = pangolin::Display("imgVideo").SetAspect(w/(float)h);
	pangolin::GlTexture texVideo(w,h,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

  pangolin::CreateDisplay()
    .SetBounds(0.0, 0.3, 0.0, 1.0)
    .SetLayout(pangolin::LayoutEqual)
    .AddDisplay(d_video);


  while( !pangolin::ShouldQuit() && running ) {    
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f,1.0f,1.0f,1.0f);

    Visualization3D_display.Activate(Visualization3D_camera);
  
    // maybe possible to draw cameras without copying to CPU?
    transformMatrix = poseToMatrix(poses);
    transformMatrix = transformMatrix.transpose(1,2);
    transformMatrix = transformMatrix.contiguous().to(torch::kCPU);

    // draw poses using OpenGL
    drawPoints();
    drawPoses();

    mtx.lock();
    if (redraw) {
      redraw = false;
      texVideo.Upload(image.data_ptr(), GL_BGR, GL_UNSIGNED_BYTE);
    }
    mtx.unlock();

    d_video.Activate();
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    texVideo.RenderToViewportFlipY();

    pangolin::FinishFrame();
  }

  // destroy OpenGL buffers
  // destroyVBO();
  running = false;

  exit(1);
}


namespace py = pybind11;

PYBIND11_MODULE(dpviewerx, m) {

  py::class_<Viewer>(m, "Viewer")
    .def(py::init<const torch::Tensor,
                  const torch::Tensor,
                  const torch::Tensor,
                  const torch::Tensor,
                  const torch::Tensor>())
    .def("update_image", &Viewer::update_image)
    .def("join", &Viewer::join);
}
