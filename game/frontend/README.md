## Getting Started

> Please note that the frontend here contains placeholder text and not the actual OMNI-EPIC website.  
> Please see [https://omni-epic.vercel.app/](https://omni-epic.vercel.app/) for the actual OMNI-EPIC website.

### Backend
In order to run the game from the root of the repository run:
```bash
cd omni_epic/
python -m game.backend.app
```

Open [http://localhost:3005](http://localhost:3005) with your browser to see the backend result.

### Frontend

Ensure that you have NodeJS installed(https://nodejs.org/en/download/)  
Ensure that you have bun installed as well(https://bun.sh)  

In order to get the packages run
```bash
cd omni_epic/game/frontend/
bun install
```

In order to run the package 
```bash
cd omni_epic/game/frontend
npm run dev
# or
bun run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the frontend result.

To edit the main page, edit /public/research-paper-content.json and the changes will be reflected in the GUI.

### Changing ports
For backend:  
In omni_epic/game/backend/app.py: `absl_app.run(lambda argv: socketio.run(app, host='0.0.0.0', port=3005))`  

For frontend:  
In omni_epic/game/frontend/.env: `NEXT_PUBLIC_API_URL=http://localhost:3005` (this should point to the backend port)  
In omni_epic/game/frontend/package.json: `"dev": "next dev"` to `"dev": "next dev -p 4000"`  
