const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');
const path = require('path');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configuration PostgreSQL
const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

// Middleware d'authentification
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) {
        return res.status(401).json({ error: 'Token manquant' });
    }
    jwt.verify(token, process.env.JWT_SECRET || 'votre_secret_jwt', (err, user) => {
        if (err) {
            return res.status(403).json({ error: 'Token invalide' });
        }
        req.user = user;
        next();
    });
};

// Routes d'authentification
app.post('/api/admin/login', (req, res) => {
    const { username, password } = req.body;
    if (username === process.env.ADMIN_USERNAME && password === process.env.ADMIN_PASSWORD) {
        const token = jwt.sign(
            { username: process.env.ADMIN_USERNAME },
            process.env.JWT_SECRET || 'votre_secret_jwt',
            { expiresIn: '1h' }
        );
        res.json({ token });
    } else {
        res.status(401).json({ error: 'Identifiants incorrects' });
    }
});

// Routes protégées par authentification
app.get('/api/admin/dashboard', authenticateToken, (req, res) => {
    res.json({ message: 'Accès autorisé au dashboard admin' });
});

// Routes principales
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public/index.html'));
});

// API pour les statistiques globales
app.get('/api/global-stats', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                SUM(confirmed) as total_confirmed,
                SUM(deaths) as total_deaths,
                SUM(recovered) as total_recovered,
                SUM(active) as total_active
            FROM daily_stats
            WHERE date = (SELECT MAX(date) FROM daily_stats)
        `);
        res.json(result.rows[0]);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour les top pays
app.get('/api/top-countries', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                c.country_name as country_region,
                d.confirmed,
                d.deaths,
                d.recovered,
                d.active,
                CAST(d.deaths AS FLOAT) / NULLIF(d.confirmed, 0) * 100 as mortality_rate
            FROM daily_stats d
            JOIN countries c ON d.country_id = c.id
            WHERE d.date = (SELECT MAX(date) FROM daily_stats)
            ORDER BY d.confirmed DESC
            LIMIT 10
        `);
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour l'évolution mondiale
app.get('/api/global-timeline', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                date,
                SUM(confirmed) as confirmed,
                SUM(deaths) as deaths,
                SUM(recovered) as recovered,
                SUM(active) as active
            FROM daily_stats
            GROUP BY date
            ORDER BY date
        `);
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour les statistiques par région WHO
app.get('/api/region-stats', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT 
                wr.region_name,
                d.date,
                SUM(d.confirmed) as confirmed,
                SUM(d.deaths) as deaths,
                SUM(d.recovered) as recovered,
                SUM(d.active) as active,
                CAST(SUM(d.deaths) AS FLOAT) / NULLIF(SUM(d.confirmed), 0) * 100 as mortality_rate
            FROM daily_stats d
            JOIN countries c ON d.country_id = c.id
            JOIN who_regions wr ON c.who_region_id = wr.id
            GROUP BY wr.region_name, d.date
            ORDER BY d.date
        `);
        res.json(result.rows);
    } catch (err) {
        console.error('Erreur détaillée:', err);
        res.status(500).json({ error: err.message });
    }
});

// API pour l'évolution d'un pays spécifique
app.get('/api/country-timeline/:country', async (req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                d.date,
                d.confirmed,
                d.deaths,
                d.recovered,
                d.active,
                CAST(d.deaths AS FLOAT) / NULLIF(d.confirmed, 0) * 100 as mortality_rate
            FROM daily_stats d
            JOIN countries c ON d.country_id = c.id
            WHERE c.country_name = $1
            ORDER BY d.date
        `, [req.params.country]);
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour la comparaison de taux entre pays
app.get('/api/country-comparison', async (req, res) => {
    const countries = req.query.countries ? req.query.countries.split(',') : [];
    if (countries.length === 0) {
        return res.status(400).json({ error: 'Aucun pays spécifié' });
    }

    try {
        const result = await pool.query(`
            SELECT
                c.country_name,
                d.date,
                d.confirmed,
                d.deaths,
                d.recovered,
                d.active,
                CAST(d.deaths AS FLOAT) / NULLIF(d.confirmed, 0) * 100 as mortality_rate
            FROM daily_stats d
            JOIN countries c ON d.country_id = c.id
            WHERE c.country_name = ANY($1)
            ORDER BY d.date
        `, [countries]);
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// API pour les données filtrées
app.get('/api/filtered-data', async (req, res) => {
    const { region, dateStart, dateEnd, minCases, maxCases } = req.query;
    let queryParams = [];
    let queryConditions = ['1=1'];

    if (region) {
        queryParams.push(region);
        queryConditions.push(`wr.region_name = $${queryParams.length}`);
    }
    if (dateStart) {
        queryParams.push(dateStart);
        queryConditions.push(`d.date >= $${queryParams.length}`);
    }
    if (dateEnd) {
        queryParams.push(dateEnd);
        queryConditions.push(`d.date <= $${queryParams.length}`);
    }
    if (minCases) {
        queryParams.push(minCases);
        queryConditions.push(`d.confirmed >= $${queryParams.length}`);
    }
    if (maxCases) {
        queryParams.push(maxCases);
        queryConditions.push(`d.confirmed <= $${queryParams.length}`);
    }

    try {
        const result = await pool.query(`
            SELECT
                c.country_name,
                wr.region_name,
                d.date,
                d.confirmed,
                d.deaths,
                d.recovered,
                d.active,
                CAST(d.deaths AS FLOAT) / NULLIF(d.confirmed, 0) * 100 as mortality_rate
            FROM daily_stats d
            JOIN countries c ON d.country_id = c.id
            JOIN who_regions wr ON c.who_region_id = wr.id
            WHERE ${queryConditions.join(' AND ')}
            ORDER BY d.confirmed DESC
        `, queryParams);
        res.json(result.rows);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Serveur démarré sur http://localhost:${PORT}`);
});