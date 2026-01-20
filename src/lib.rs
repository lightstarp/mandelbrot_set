use std::{iter, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, Fullscreen},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ]
        }
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

#[repr(C)]
#[derive(Debug, Clone)]
struct Camera {
    position: cgmath::Vector2<f32>,
    zoom: f32,
}
impl Camera {
    fn new() -> Camera {
        Camera {
            position: cgmath::Vector2::new(0.0, 0.0),
            zoom: 0.2,
        }
    }
}
struct CameraController {
    is_pressed_left:  bool,
    is_pressed_right: bool,
    is_pressed_up:    bool,
    is_pressed_down:  bool,
    is_pressed_zoom_in:  bool,
    is_pressed_zoom_out: bool,
}
impl CameraController {
    fn new() -> CameraController {
        CameraController {
            is_pressed_left:  false,
            is_pressed_right: false,
            is_pressed_up:    false,
            is_pressed_down:  false,
            is_pressed_zoom_in:  false,
            is_pressed_zoom_out: false,
        }
    }
    fn handle_key(&mut self, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::ArrowLeft,  press) => self.is_pressed_left  = press,  
            (KeyCode::ArrowRight, press) => self.is_pressed_right = press,
            (KeyCode::ArrowUp,    press) => self.is_pressed_up    = press,
            (KeyCode::ArrowDown,  press) => self.is_pressed_down  = press,
            (KeyCode::KeyZ,       press) => self.is_pressed_zoom_in  = press,
            (KeyCode::KeyX,       press) => self.is_pressed_zoom_out = press,
            _ => {}
        }
    }
    const MOVE_SPEED: f32 = 0.500;
    const ZOOM_SPEED: f32 = 2.000;
    fn update(
        &mut self,
        camera: &mut Camera,
        delta_time: f32,
    ) {
        use cgmath::Vector2;

        let move_speed = Self::MOVE_SPEED * delta_time / camera.zoom;
        let zoom_speed = Self::ZOOM_SPEED.powf(delta_time);

        if self.is_pressed_left  { camera.position -= Vector2::new(move_speed, 0.0) }
        if self.is_pressed_right { camera.position += Vector2::new(move_speed, 0.0) }
        if self.is_pressed_up    { camera.position += Vector2::new(0.0, move_speed) }
        if self.is_pressed_down  { camera.position -= Vector2::new(0.0, move_speed) }
        if self.is_pressed_zoom_in  { camera.zoom *= zoom_speed }
        if self.is_pressed_zoom_out { camera.zoom /= zoom_speed }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    pos: [f32; 2],
    zoom: f32,
    width: f32,
    height: f32,
    padding: f32,
}
impl CameraUniform {
    fn new() -> Self {
        Self {
            pos: cgmath::Vector2::new(0.0, 0.0).into(),
            zoom: 0.5,
            width: 100.0,
            height: 100.0,
            padding: 0.0,
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, width: f32, height: f32) {
        self.pos = camera.position.into();
        self.zoom = camera.zoom;
        self.width = width;
        self.height = height;
    }
}

struct Report {
    elapsed: f32,
    fps_space: u32,
    fps_buf: f32,
    fps_buf_count: u32,
    fps_data: Vec<f32>, // 1second
}
impl Report {
    const FPS_CAPACITY: usize = 100;
    fn new() -> Report {
        Report {
            elapsed: 0.0,
            fps_space: 1,
            fps_buf: 0.0,
            fps_buf_count: 0,
            fps_data: Vec::with_capacity(Self::FPS_CAPACITY),
        }
    }
    fn update(
        &mut self,
        delta_time: f32,
    ) {
        self.elapsed += delta_time;
        if self.elapsed >= 1.0 {
            self.elapsed = 0.0;
            self.update_intarval(delta_time);
        }
    }
    fn update_intarval(
        &mut self,
        delta_time: f32,
    ) {
        self.fps_buf += 1.0 / delta_time;
        self.fps_buf_count += 1;
        if self.fps_buf_count >= self.fps_space {
            if self.fps_data.len() >= Self::FPS_CAPACITY {
                self.fps_space *= 2;
                for i in 0..Self::FPS_CAPACITY/2 {
                    self.fps_data[i] = self.fps_data[i*2] + self.fps_data[i*2+1]
                }
                unsafe { self.fps_data.set_len(Self::FPS_CAPACITY/2) }
            } else {
                self.fps_data.push(self.fps_buf);
                self.fps_buf = 0.0;
            }
        }
    }
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer, 
    num_indices: u32,

    window: Arc<Window>,

    config_data: ConfigData,

    fps_interval: Instant,
    delta_time: f32,

    report: Report,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl State {
    async fn new(window: Arc<Window>, config_data: ConfigData) -> anyhow::Result<State> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };

        // vvv ########## Create camera_w ########## vvv
        let camera = Camera::new();
        let camera_controller = CameraController::new();
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, config.width as f32, config.height as f32);
 
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("CameraW Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_w_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_w_bind_group"),
        });
        // ^^^ ########## Create camera_w ########## ^^^

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(&format!("fs_{}ex", config_data.execution)),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let vartices: &[Vertex] = &[
            Vertex { position: [-1.0, -1.0] },
            Vertex { position: [ 1.0, -1.0] },
            Vertex { position: [-1.0,  1.0] },
            Vertex { position: [ 1.0,  1.0] },
        ];
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vartices),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let indices: &[u16] = &[
            0, 1, 2,
            2, 1, 3,
        ];
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }
        );
        let num_indices = indices.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,

            window,

            config_data,

            fps_interval: Instant::now(),
            delta_time: 1.0,
            report: Report::new(),

            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    fn update(&mut self) {
        let _ = self.config_data;

        self.delta_time = self.fps_interval.elapsed().as_secs_f32();
        self.fps_interval = Instant::now();
        
        self.report.update(self.delta_time);

        self.camera_controller.update(&mut self.camera, self.delta_time);
        self.camera_uniform.update_view_proj(&mut self.camera, self.config.width as f32, self.config.height as f32);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.03,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1 as _);
        }
        

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => {
                event_loop.exit();
            },
            (KeyCode::F11   , true) => {
                if self.window.fullscreen().is_some() {
                    self.window.set_fullscreen(None);
                } else {
                    self.window.current_monitor().map(|monitor| {
                        monitor.video_modes().next().map(|video_mode| {
                            if cfg!(any(target_os = "macos", unix)) {
                                self.window.set_fullscreen(Some(Fullscreen::Borderless(Some(monitor))));
                            } else {
                                self.window.set_fullscreen(Some(Fullscreen::Exclusive(video_mode)));
                            }
                        })
                    });
                }
            },
            _ => {
                self.camera_controller.handle_key(code, is_pressed)
            }
        }
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    config_data: ConfigData,
    state: Option<State>,
}

impl App {
    pub fn new(config_data: ConfigData, #[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            config_data,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes().with_title("mandelbrot set");

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // If we are not on web we can use pollster to
            // await the
            self.state = Some(pollster::block_on(State::new(window, self.config_data.clone())).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(proxy
                        .send_event(
                            State::new(window)
                                .await
                                .expect("Unable to create canvas!!!")
                        )
                        .is_ok())
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(..) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match (button, state.is_pressed()) {
                (MouseButton::Left, true) => {}
                (MouseButton::Left, false) => {}
                _ => {}
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let config_data = {
        #[cfg(not(target_arch = "wasm32"))]
        match load_config() {
            Ok(t) => t,
            Err(()) => ConfigData::new(),
        }
        #[cfg(target_arch = "wasm32")]
        ConfigData::new()
    };

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        config_data,
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    
    #[cfg(not(target_arch = "wasm32"))]
    {
        if app.config_data.create_report {
            create_report(&app);
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct ConfigData {
    execution:     u32,
    create_report: bool,
}
impl ConfigData {
    fn new() -> ConfigData {
        ConfigData {
            execution: 20,
            create_report: true,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_config() -> Result<ConfigData, ()> {
    use std::{fs, fs::File, io::{Write, Read}, str::Chars};

    println!("--- Config Loading status ---");
    // Load File
    match fs::create_dir_all("config") {
        Ok(()) => {}
        Err(e) => {
            println!("Status: Error");
            println!("{:?}", e.kind());
            return Err(())
        }
    }

    match File::create_new("config/config.txt") {
        Ok(mut t) => match t.write_all("execution = 20;\ncreate_report = true;".as_bytes()) {
            Ok(()) => {}
            Err(e) => {
                println!("Status: Error");
                println!("{:?}", e.kind());
                return Err(())
            }
        }
        Err(..) => {}
    }

    let mut file = match File::open("config/config.txt"){
        Ok(t) => t,
        Err(e) => {
            println!("Status: Error");
            println!("{:?}", e.kind());
            return Err(())
        }
    };

    let mut buf = String::new();
    match file.read_to_string(&mut buf) {
        Ok(..) => {}
        Err(e) => {
            println!("Status: Error");
            println!("{:?}", e.kind());
            return Err(())
        }
    }

    struct Parser<'r> {
        chars: Chars<'r>,
        peek: Option<char>,
    }
    impl<'r> Parser<'r> {
        #[inline]
        fn next(&mut self) {
            self.peek = self.chars.next();
        }
        fn skip_space(&mut self) -> &mut Parser<'r> {
            loop {
                match self.peek {
                    Some(c) => if !c.is_whitespace() {
                        break
                    }
                    _ => break,
                }
                self.next();
            }
            self
        }
        fn read_ident(&mut self) -> String {
            let mut collect = String::new();
            loop {
                match self.peek {
                    Some(c @ ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) => collect.push(c),
                    _ => break,
                }
                self.next();
            }
            collect
        }
    }

    // Lexer
    let mut p = Parser {
        chars: buf.chars(),
        peek: None,
    };
    p.next();

    let mut token: Vec<(String, String)> = Vec::new();

    let mut parameter_count = 1;
    loop {
        let ident = p.skip_space().read_ident();
        if p.skip_space().peek == Some('=') {
            p.next();
        } else {
            println!("Status: SyntaxError");
            let th = match parameter_count {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            };
            println!("The {parameter_count}{th} parameter was missing the required \"=\" in the correct position");
            println!("The format must be in the form \"[ident] = [value];\"");
            return Err(())
        }
        let value = p.skip_space().read_ident();
        if p.skip_space().peek == Some(';') {
            p.next();
        } else {
            println!("Status: SyntaxError");
            let th = match parameter_count {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            };
            println!("The {parameter_count}{th} parameter was missing the required \";\" in the correct position");
            println!("The format must be in the form \"[ident] = [value];\"");
            return Err(())
        }
        token.push((ident, value));
        if p.skip_space().peek == None {
            break
        }
        parameter_count += 1;
    }

    let mut config_data = ConfigData::new();

    for (ident, value) in token.iter() {
        if ident == "execution" {
            config_data.execution = match value.parse::<u32>() {
                Ok(t) => match t {
                    10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 => t,
                    _ => {
                        println!("Status: SyntaxError");
                        println!("The value of parameter \"execution\" must 10 | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000");
                        return Err(())
                    }
                },
                Err(..) => {
                    println!("Status: SyntaxError");
                    println!("The value of parameter \"execution\" is not integer");
                    return Err(())
                }
            };
            continue;
        }
        if ident == "create_report" {
            config_data.create_report = match value.parse::<bool>() {
                Ok(t) => t,
                Err(..) => {
                    println!("Status: Error");
                    println!("The value of parameter \"create_report\" is not bool");
                    println!("it must true | false");
                    return Err(())
                }
            };
            continue;
        }
        // The parameter “create_report” does not exist.
        println!("Status: Error");
        println!("The parameter \"{ident}\" does not exist.");
        return Err(())
    };

    println!("Status: Ok");
    Ok(config_data)
}

#[cfg(not(target_arch = "wasm32"))]
fn create_report(app: &App) {
    use std::{fs, fs::File, io::{Write, ErrorKind}};

    println!("--- Report creation status ---");

    let state = match &app.state {
        Some(t) => t,
        None => {
            println!("Status: Error");
            println!("The report could not be created because no data existed");
            return
        }
    };

    let time = &chrono::Utc::now().format("%Y-%m-%d-%H-%M-%S").to_string();
    let mut file = match File::create_new(format!("report/{time}.txt")) {
        Ok(f) => f,
        Err(e) => {
            match e.kind() {
                ErrorKind::NotFound => {
                    match fs::create_dir("report") {
                        Ok(..) => (),
                        Err(e) => {
                            println!("Status: Error");
                            println!("{:?}", e.kind());
                            return
                        },
                    };
                    match File::create(format!("report/{time}.txt")) {
                        Ok(f) => f,
                        Err(e) => {
                            println!("Status: Error");
                            println!("{:?}", e.kind());
                            return
                        },
                    }
                }
                e => {
                    println!("Status: Error");
                    println!("{:?}", e);
                    return
                }
            }
        }
    };
    
    let mut s = String::new();
    
    s.push_str(&format!("### Config data ###\n"));
    s.push_str(&format!("execution = {};\n"    , app.config_data.execution));
    s.push_str(&format!("create_report = {};\n", app.config_data.create_report));
    
    s.push_str("\n");
    s.push_str(&format!("### Fps data ###\n"));

    if state.report.fps_data.len() == 0 {
        s.push_str("No data available");
    } else {
        let fps_average = state.report.fps_data.iter().sum::<f32>() / state.report.fps_data.len() as f32;
        s.push_str(&format!("The average fps was {:>14.6}\n", fps_average));
        let mut vec = state.report.fps_data.clone();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let fps_median = vec[vec.len() / 2];
        s.push_str(&format!("The median  fps was {:>14.6}\n", fps_median));
    
        s.push_str("\n");
        for (fps, i) in state.report.fps_data.iter().zip(0..) {
            s.push_str(&format!("{:>6}seconds: {:>14.6}fps\n", i * state.report.fps_space, fps));
        }
    }
    match file.write_all(s.as_bytes()) {
        Ok(()) => {}
        Err(e) => {
            println!("Status: Error");
            println!("{:?}", e.kind());
            return
        }
    }

    println!("Status: Ok");
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
