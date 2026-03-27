use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Color,
    widgets::{
        Widget,
        canvas::{Canvas, Points},
    },
};

use super::worldmap;

/// A spinning ASCII globe rendered using braille unicode characters.
pub struct Globe {
    /// Current rotation angle around the Y axis (radians).
    angle: f64,
    /// Rotation speed in radians per tick.
    rotation_speed: f64,
    /// 3D cartesian coordinates of continent points on the unit sphere.
    continent_points: Vec<(f64, f64, f64)>,
    /// 3D coordinates of active network nodes.
    node_positions: Vec<(f64, f64, f64)>,
    /// This node's position (if known).
    my_position: Option<(f64, f64, f64)>,
    /// Throughput sparkline history for subtle pulsing.
    pulse_phase: f64,
}

impl Globe {
    pub fn new() -> Self {
        let raw_points = worldmap::get_world_points();
        let continent_points: Vec<(f64, f64, f64)> =
            raw_points.iter().map(|&(lat, lon)| latlon_to_xyz(lat, lon)).collect();

        Self {
            angle: 0.0,
            rotation_speed: 0.02, // ~1 revolution per 314 ticks (~31s at 10fps)
            continent_points,
            node_positions: Vec::new(),
            my_position: None,
            pulse_phase: 0.0,
        }
    }

    /// Advance the globe rotation by one tick.
    pub fn tick(&mut self) {
        self.angle += self.rotation_speed;
        if self.angle > std::f64::consts::TAU {
            self.angle -= std::f64::consts::TAU;
        }
        self.pulse_phase += 0.1;
    }

    /// Set the rotation angle directly (for startup animation).
    #[allow(dead_code)]
    pub fn set_angle(&mut self, angle: f64) {
        self.angle = angle;
    }

    /// Set mock node positions for demo purposes.
    pub fn set_mock_nodes(&mut self) {
        // Scatter some nodes around the world
        let node_coords = [
            (37.7749, -122.4194), // San Francisco
            (40.7128, -74.0060),  // New York
            (51.5074, -0.1278),   // London
            (48.8566, 2.3522),    // Paris
            (35.6762, 139.6503),  // Tokyo
            (1.3521, 103.8198),   // Singapore
            (-33.8688, 151.2093), // Sydney
            (55.7558, 37.6173),   // Moscow
            (19.4326, -99.1332),  // Mexico City
            (-23.5505, -46.6333), // Sao Paulo
            (25.2048, 55.2708),   // Dubai
            (37.5665, 126.9780),  // Seoul
            (52.5200, 13.4050),   // Berlin
            (43.6532, -79.3832),  // Toronto
            (-1.2921, 36.8219),   // Nairobi
        ];

        self.node_positions =
            node_coords.iter().map(|&(lat, lon)| latlon_to_xyz(lat, lon)).collect();

        // User's node (San Francisco for demo)
        self.my_position = Some(latlon_to_xyz(37.7749, -122.4194));
    }

    /// Project a 3D point with the current rotation, returning 2D coords if visible.
    fn project(&self, point: (f64, f64, f64)) -> Option<(f64, f64)> {
        let (x, y, z) = point;
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        // Y-axis rotation
        let rx = x * cos_a - z * sin_a;
        let rz = x * sin_a + z * cos_a;
        let ry = y;

        // Only render front-facing points
        if rz > -0.1 {
            // Slight depth perspective
            let scale = 1.0 + rz * 0.15;
            Some((rx * scale, ry * scale))
        } else {
            None
        }
    }

    /// Render the globe into a ratatui Canvas widget.
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let visible_continents: Vec<(f64, f64)> =
            self.continent_points.iter().filter_map(|&p| self.project(p)).collect();

        let visible_nodes: Vec<(f64, f64)> =
            self.node_positions.iter().filter_map(|&p| self.project(p)).collect();

        let visible_me: Vec<(f64, f64)> =
            self.my_position.iter().filter_map(|&p| self.project(p)).collect();

        // Generate sphere outline (circle)
        let outline: Vec<(f64, f64)> = (0..120)
            .map(|i| {
                let theta = (i as f64 / 120.0) * std::f64::consts::TAU;
                (theta.cos(), theta.sin())
            })
            .collect();

        let canvas = Canvas::default()
            .x_bounds([-1.3, 1.3])
            .y_bounds([-1.3, 1.3])
            .paint(move |ctx| {
                // Draw sphere outline
                ctx.draw(&Points { coords: &outline, color: Color::DarkGray });

                // Draw continents
                ctx.draw(&Points { coords: &visible_continents, color: Color::White });

                // Draw network nodes
                ctx.draw(&Points { coords: &visible_nodes, color: Color::Green });

                // Draw "you" marker
                ctx.draw(&Points { coords: &visible_me, color: Color::Yellow });
            })
            .marker(ratatui::symbols::Marker::Braille);

        canvas.render(area, buf);
    }
}

/// Convert latitude/longitude (degrees) to 3D cartesian coordinates on a unit sphere.
fn latlon_to_xyz(lat_deg: f64, lon_deg: f64) -> (f64, f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let x = lat.cos() * lon.cos();
    let y = lat.sin();
    let z = lat.cos() * lon.sin();
    (x, y, z)
}
